package gov.irs.directfile.stateapi;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.Security;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.Period;
import java.util.*;
import java.util.concurrent.CompletableFuture;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.client.RestTemplate;
import software.amazon.awssdk.services.s3.model.DeleteObjectResponse;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.stateapi.configuration.S3ConfigurationProperties;
import gov.irs.directfile.stateapi.encryption.JwtSigner;
import gov.irs.directfile.stateapi.model.AuthCodeRequest;
import gov.irs.directfile.stateapi.model.AuthorizationCode;
import gov.irs.directfile.stateapi.model.ExportResponse;
import gov.irs.directfile.stateapi.model.TaxReturnToExport;
import gov.irs.directfile.stateapi.repository.AuthorizationCodeRepository;

import static gov.irs.directfile.stateapi.encryption.Decryptor.aesGcmDecrypt;
import static gov.irs.directfile.stateapi.encryption.Decryptor.rsaDecryptWithPrivateKey;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.collection.IsIterableContainingInAnyOrder.containsInAnyOrder;
import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT,
        properties = {"server.port=8081"})
@ActiveProfiles({"development", "test", "integration-test"})
@Slf4j
@SuppressWarnings("null")
@EnabledIfSystemProperty(named = "runIntegrationTests", matches = "true")
public class StateApiAppTest {
    private static final String PRIVATE_KEY_PATH = "src/test/resources/certificates/fakestate.key";
    private static final String SUBMISSION_ID = "someSubmissionId";
    private static final String TAX_YEAR = "2022";
    private static final String TAX_RETURN_ID = "ae019609-99e0-4ef5-85bb-ad90dc302e70";
    private static final String XML_OBJECT_KEY =
            TAX_YEAR + "/taxreturns/" + TAX_RETURN_ID + "/submissions/" + SUBMISSION_ID + ".xml";
    private static final String AUTHORIZATION_CODE_URL = "http://localhost:8081/state-api/authorization-code";
    private static final String EXPORT_RETURN_URL = "http://localhost:8081/state-api/export-return";
    private final RestTemplate restTemplate = new RestTemplate();
    private final HttpHeaders headers = new HttpHeaders();

    @Autowired
    AuthorizationCodeRepository authorizationCodeRepository;

    @Autowired
    S3AsyncEncryptionClient s3AsyncEncryptionClient;

    @Autowired
    S3ConfigurationProperties s3ConfigurationProperties;

    @BeforeEach
    public void setUp() {
        Security.addProvider(new BouncyCastleProvider());
        headers.setContentType(MediaType.APPLICATION_JSON);
    }

    private void putEncryptedXml() {
        Path resourceDir = Paths.get("src/test/resources/xml/sample-export.xml");
        // Call putObject to encrypt the object and upload it to S3
        CompletableFuture<PutObjectResponse> futurePut = s3AsyncEncryptionClient.putObject(
                builder -> builder.bucket(s3ConfigurationProperties.getTaxReturnXmlBucketName())
                        .key(XML_OBJECT_KEY)
                        .build(),
                resourceDir);
        // Block on completion of the futurePut
        futurePut.join();
    }

    private void deleteEncryptedXml() {
        CompletableFuture<DeleteObjectResponse> futureDelete = s3AsyncEncryptionClient.deleteObject(
                builder -> builder.bucket(s3ConfigurationProperties.getTaxReturnXmlBucketName())
                        .key(XML_OBJECT_KEY)
                        .build());
        futureDelete.join();
    }

    private HttpEntity<AuthCodeRequest> createAuthCodeRequestBodyAndEntity(String uuid, int taxYear, String stateCode) {
        var tin = "213456789";

        AuthCodeRequest requestObject =
                new AuthCodeRequest(UUID.fromString(uuid), tin, taxYear, stateCode, SUBMISSION_ID);

        return new HttpEntity<>(requestObject, headers);
    }

    @Test
    public void integrationTests_successfulResponse() throws Exception {
        putEncryptedXml();
        var authCodeRequestEntity = createAuthCodeRequestBodyAndEntity(TAX_RETURN_ID, Integer.parseInt(TAX_YEAR), "FS");
        ResponseEntity<AuthCodeResponse> authCodeResponse =
                restTemplate.postForEntity(AUTHORIZATION_CODE_URL, authCodeRequestEntity, AuthCodeResponse.class);

        UUID ac = authCodeResponse.getBody().getAuthCode();

        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"123456\",\"sub\":\"" + ac.toString() + "\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);

        // Set the desired header(s)
        headers.set("Authorization", "Bearer " + jwtToken);

        var exportReturnRequestEntity = new HttpEntity<>(headers);
        var exportResponse =
                restTemplate.exchange(EXPORT_RETURN_URL, HttpMethod.GET, exportReturnRequestEntity, String.class);
        if (Objects.requireNonNull(exportResponse.getBody()).contains("error"))
            throw new AssertionError("Integration tests failed unexpectedly!");

        String encodedSecret = exportResponse.getHeaders().get("SESSION-KEY").getFirst();
        String encodedIV =
                exportResponse.getHeaders().get("INITIALIZATION-VECTOR").getFirst();
        String encodedAuthenticationTag =
                exportResponse.getHeaders().get("AUTHENTICATION-TAG").getFirst();

        String taxReturn = exportResponse.getBody();
        ObjectMapper om = new ObjectMapper();
        ExportResponse response = om.readValue(taxReturn.getBytes(), ExportResponse.class);

        if (!response.getStatus().equals("success")) throw new AssertionError("Integration tests failed unexpectedly!");

        byte[] secret = rsaDecryptWithPrivateKey(Base64.getDecoder().decode(encodedSecret), PRIVATE_KEY_PATH);

        // decrypt the data using the secret and iv
        byte[] decryptedBytes = aesGcmDecrypt(
                Base64.getDecoder().decode(response.getTaxReturn()),
                secret,
                Base64.getDecoder().decode(encodedIV),
                Base64.getDecoder().decode(encodedAuthenticationTag));

        TaxReturnToExport taxReturnData = om.readValue(decryptedBytes, TaxReturnToExport.class);

        assertEquals(taxReturnData.getStatus(), "accepted");
        assertEquals(taxReturnData.getSubmissionId(), SUBMISSION_ID);
        assertNotNull(taxReturnData.getXml());
        assertFalse(taxReturnData.getDirectFileData().isEmpty());
        assertTrue(taxReturnData.getDirectFileData().containsKey("familyAndHousehold"));

        var familyAndHousehold =
                (List<Map<String, Object>>) taxReturnData.getDirectFileData().get("familyAndHousehold");
        var expectedPerson = new HashMap<>();
        expectedPerson.put("firstName", "Sammy");
        expectedPerson.put("middleInitial", null);
        expectedPerson.put("lastName", "Smith");
        expectedPerson.put("suffix", "I");
        expectedPerson.put("dateOfBirth", "2013-01-21");

        expectedPerson.put("tin", "200-01-1234");
        expectedPerson.put("relationship", "biologicalChild");
        expectedPerson.put("isClaimedDependent", true);
        expectedPerson.put("eligibleDependent", true);
        expectedPerson.put("residencyDuration", "allYear");
        expectedPerson.put("monthsLivedWithTPInUS", "twelve");
        expectedPerson.put("ssnNotValidForEmployment", false);
        expectedPerson.put("qualifyingChild", true);
        expectedPerson.put("hohQualifyingPerson", true);
        expectedPerson.put("scheduleEicLine4bYes", false);
        expectedPerson.put("scheduleEicLine4aYes", true);
        expectedPerson.put("scheduleEicLine4aNo", false);

        assertEquals(1, familyAndHousehold.size());
        assertEquals(expectedPerson, familyAndHousehold.getFirst());

        var filers =
                (List<Map<String, Object>>) taxReturnData.getDirectFileData().get("filers");

        var expectedFiler1 = new HashMap<>();
        expectedFiler1.put("firstName", "Samuel");
        expectedFiler1.put("middleInitial", null);
        expectedFiler1.put("lastName", "Smith");
        expectedFiler1.put("suffix", "Jr");
        expectedFiler1.put("dateOfBirth", "1985-09-29");
        expectedFiler1.put("tin", "100-01-1234");
        expectedFiler1.put("isPrimaryFiler", true);
        expectedFiler1.put("ssnNotValidForEmployment", false);
        expectedFiler1.put("educatorExpenses", "200.00");
        expectedFiler1.put("hsaTotalDeductibleAmount", "600.00");
        expectedFiler1.put("isDisabled", false);
        expectedFiler1.put("isStudent", false);
        expectedFiler1.put("interestReportsTotal", "3000.00");
        expectedFiler1.put("form1099GsTotal", "15000.00");

        var expectedFiler2 = new HashMap<>();
        expectedFiler2.put("firstName", "Judy");
        expectedFiler2.put("middleInitial", null);
        expectedFiler2.put("lastName", "Johnson");
        expectedFiler2.put("suffix", null);
        expectedFiler2.put("dateOfBirth", "1985-10-18");
        expectedFiler2.put("tin", "100-02-1234");
        expectedFiler2.put("isPrimaryFiler", false);
        expectedFiler2.put("ssnNotValidForEmployment", false);
        expectedFiler2.put("educatorExpenses", "100.00");
        expectedFiler2.put("hsaTotalDeductibleAmount", "500.00");
        expectedFiler2.put("isDisabled", false);
        expectedFiler2.put("isStudent", false);
        expectedFiler2.put("interestReportsTotal", "300.00");
        expectedFiler2.put("form1099GsTotal", "1500.00");
        assertEquals(2, filers.size());
        assertThat(filers, containsInAnyOrder(expectedFiler1, expectedFiler2));

        var intReports =
                (List<Map<String, Object>>) taxReturnData.getDirectFileData().get("interestReports");
        var intRpt = new HashMap<String, Object>();
        intRpt.put("has1099", true);
        intRpt.put("1099Amount", "800.00");
        intRpt.put("interestOnGovernmentBonds", "300");
        intRpt.put("taxExemptInterest", "200.00");
        intRpt.put("recipientTin", "123-45-6789");
        intRpt.put("no1099Amount", null);
        intRpt.put("payer", "JPM Bank");
        intRpt.put("payerTin", "01-1234567");
        intRpt.put("taxWithheld", "120");
        intRpt.put("taxExemptAndTaxCreditBondCusipNo", "01234567A");
        assertEquals(1, intReports.size());
        assertEquals(intRpt, intReports.getFirst());

        var form1099Gs =
                (List<Map<String, Object>>) taxReturnData.getDirectFileData().get("form1099Gs");
        var form1099G = new HashMap<String, Object>();
        form1099G.put("has1099", true);
        form1099G.put("recipientTin", "123-45-6789");
        form1099G.put("payer", "State of california");
        form1099G.put("payerTin", "321-54-9876");
        form1099G.put("amount", "100.00");
        form1099G.put("federalTaxWithheld", "20.00");
        form1099G.put("stateIdNumber", "123456");
        form1099G.put("stateTaxWithheld", "10.00");
        form1099G.put("amountPaidBackForBenefitsInTaxYear", "25.00");
        assertEquals(1, form1099Gs.size());
        assertEquals(form1099G, form1099Gs.getFirst());

        // socialSecurityReports
        var ssRpts =
                (List<Map<String, Object>>) taxReturnData.getDirectFileData().get("socialSecurityReports");
        var ssRpt = new HashMap<String, Object>();
        ssRpt.put("recipientTin", "123-45-6789");
        ssRpt.put("netBenefits", "21000.00");
        ssRpt.put("formType", "SSA-1099");
        assertEquals(1, ssRpts.size());
        assertEquals(ssRpt, ssRpts.getFirst());

        var formW2s =
                (List<Map<String, Object>>) taxReturnData.getDirectFileData().get("formW2s");
        assertEquals(1, formW2s.size());
        var expectedW2 = new HashMap<String, Object>();
        expectedW2.put("unionDuesAmount", "40.00");
        expectedW2.put("BOX14_NJ_UIHCWD", "101.00");
        expectedW2.put("BOX14_NJ_UIWFSWF", "101.00");
        assertEquals(expectedW2, formW2s.getFirst());

        deleteEncryptedXml();
        log.info("Integration tests succeeded");
    }

    @Test
    public void integrationTests_AccountIdNotExist() throws Exception {

        String ac = "d6e34be4-11df-4b5e-808f-bc48a9f4870b";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"12345\",\"sub\":\"" + ac + "\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);

        // Set the desired header(s)
        headers.set("account-id", "12345");
        headers.set("Authorization", "Bearer " + jwtToken);
        var exportReturnRequestEntity = new HttpEntity<>(headers);

        var response = restTemplate.exchange(
                EXPORT_RETURN_URL, HttpMethod.GET, exportReturnRequestEntity, ExportResponse.class);

        ExportResponse responseBody = response.getBody();

        assertNotNull(responseBody);
        assertEquals("error", responseBody.getStatus());
        assertEquals("E_ACCOUNT_ID_NOT_EXIST", responseBody.getError());

        log.info("Test succeeded with expected error: E_ACCOUNT_ID_NOT_EXIST");
    }

    @Test
    public void integrationTests_BearerTokenMissing() throws Exception {
        String ac = "d6e34be4-11df-4b5e-808f-bc48a9f4870b";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"123456\",\"sub\":\"" + ac + "\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);
        // Set the desired header(s)
        headers.set("account-id", "123456");
        headers.set("Authorization", "NoBearer " + jwtToken);

        var exportRequest = new HttpEntity<>(headers);
        var exportResponse = restTemplate.exchange(EXPORT_RETURN_URL, HttpMethod.GET, exportRequest, String.class);

        String responseBody = exportResponse.getBody();

        assertNotNull(responseBody);
        assertTrue(responseBody.startsWith("{\"status\":\"error\""));
        assertTrue(responseBody.contains("E_BEARER_TOKEN_MISSING"));

        log.info("Test succeeded with expected error: E_BEARER_TOKEN_MISSING");
    }

    @Test
    public void integrationTests_AuthCodeExpired() throws Exception {
        String ac = "2511725b-094f-4d1f-a7f7-da6003a5f8e7";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"123456\",\"sub\":\"" + ac + "\",\"iat\":1516239022}";

        // add test data if not exists
        AuthorizationCode acObject = new AuthorizationCode();
        acObject.setAuthorizationCode(UUID.fromString(ac));
        var existingAuthorizationCodeObject = authorizationCodeRepository
                .getByAuthorizationCode(acObject.getAuthorizationCode())
                .block();

        if (existingAuthorizationCodeObject == null) {
            log.info("Adding test case data");
            acObject.setTaxReturnUuid(UUID.randomUUID());
            acObject.setTaxYear(2022);
            Timestamp expireAt = Timestamp.from(Instant.now().minus(Period.ofDays(30)));
            acObject.setExpiresAt(expireAt);
            acObject.setStateCode("FS");
            acObject.setSubmissionId("someSubmissionId");

            authorizationCodeRepository.save(acObject).block();
        } else {
            log.info("Test case data already exists: {}", existingAuthorizationCodeObject);
        }

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);
        // Set the desired header(s)
        headers.set("account-id", "123456");
        headers.set("Authorization", "Bearer " + jwtToken);

        var exportRequest = new HttpEntity<>(headers);
        var exportResponse = restTemplate.exchange(EXPORT_RETURN_URL, HttpMethod.GET, exportRequest, String.class);

        String responseBody = exportResponse.getBody();

        assertNotNull(responseBody);
        assertTrue(responseBody.startsWith("{\"status\":\"error\""));
        assertTrue(responseBody.contains("E_AUTHORIZATION_CODE_EXPIRED"));

        log.info("Test succeeded with expected error: E_AUTHORIZATION_CODE_EXPIRED");
    }

    @Test
    public void integrationTests_AuthCodeInvalidFormat() throws Exception {
        String ac = "11df-4b5e-808f-bc48a9f4870b";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"123456\",\"sub\":\"" + ac + "\",\"iat\":1516239022}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);
        // Set the desired header(s)
        headers.set("account-id", "123456");
        headers.set("Authorization", "Bearer " + jwtToken);

        var exportRequestEntity = new HttpEntity<>(headers);
        var exportResponse =
                restTemplate.exchange(EXPORT_RETURN_URL, HttpMethod.GET, exportRequestEntity, ExportResponse.class);

        ExportResponse responseBody = exportResponse.getBody();

        assertNotNull(responseBody);
        assertEquals("error", responseBody.getStatus());
        assertEquals("E_AUTHORIZATION_CODE_INVALID_FORMAT", responseBody.getError());

        log.info("Test succeeded with expected error: E_AUTHORIZATION_CODE_INVALID_FORMAT");
    }

    @Test
    public void integrationTests_AuthCodeNotExist() throws Exception {
        String ac = "d6e34be4-11df-4b5e-808f-bc48a9f4870c";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"123456\",\"sub\":\"" + ac + "\",\"iat\":1516239022}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);
        // Set the desired header(s)
        headers.set("account-id", "123456");
        headers.set("Authorization", "Bearer " + jwtToken);

        var exportRequestEntity = new HttpEntity<>(headers);
        var exportResponse =
                restTemplate.exchange(EXPORT_RETURN_URL, HttpMethod.GET, exportRequestEntity, ExportResponse.class);

        ExportResponse responseBody = exportResponse.getBody();

        assertNotNull(responseBody);
        assertEquals("error", responseBody.getStatus());
        assertEquals("E_AUTHORIZATION_CODE_NOT_EXIST", responseBody.getError());

        log.info("Test succeeded with expected error: E_AUTHORIZATION_CODE_NOT_EXIST");
    }

    @Test
    public void integrationTests_JWTVerificationFailed() throws Exception {

        String ac = "d6e34be4-11df-4b5e-808f-bc48a9f4870c";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"123456\",\"sub\":\"" + ac + "\",\"iat\":1516239022}";
        var badPrivateKeyPath = "src/test/resources/certificates/fakestate_bad.key";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, badPrivateKeyPath);
        // Set the desired header(s)
        headers.set("account-id", "123456");
        headers.set("Authorization", "Bearer " + jwtToken);

        var exportRequestEntity = new HttpEntity<>(headers);
        var exportResponse =
                restTemplate.exchange(EXPORT_RETURN_URL, HttpMethod.GET, exportRequestEntity, ExportResponse.class);

        ExportResponse responseBody = exportResponse.getBody();

        assertNotNull(responseBody);
        assertEquals("error", responseBody.getStatus());
        assertEquals("E_JWT_VERIFICATION_FAILED", responseBody.getError());

        log.info("Test succeeded with expected error: E_JWT_VERIFICATION_FAILED");
    }
}
