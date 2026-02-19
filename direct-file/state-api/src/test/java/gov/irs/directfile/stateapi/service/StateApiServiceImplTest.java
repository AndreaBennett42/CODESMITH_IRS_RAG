package gov.irs.directfile.stateapi.service;

import java.io.FileInputStream;
import java.io.InputStream;
import java.security.Security;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.sql.Timestamp;
import java.util.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbusds.jose.shaded.gson.Gson;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.authorization.AuthorizationTokenClaims;
import gov.irs.directfile.stateapi.authorization.AuthorizationTokenService;
import gov.irs.directfile.stateapi.configuration.CertificationOverrideProperties;
import gov.irs.directfile.stateapi.dto.StateProfileDTO;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.exception.StateNotExistException;
import gov.irs.directfile.stateapi.model.*;
import gov.irs.directfile.stateapi.repository.AuthorizationCodeRepository;
import gov.irs.directfile.stateapi.repository.DirectFileClient;
import gov.irs.directfile.stateapi.repository.StateApiS3Client;
import gov.irs.directfile.stateapi.repository.StateProfileRepository;
import gov.irs.directfile.stateapi.repository.facts.ExportedFactsClient;

import static gov.irs.directfile.stateapi.encryption.Decryptor.aesGcmDecrypt;
import static gov.irs.directfile.stateapi.encryption.Decryptor.rsaDecryptWithPrivateKey;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class StateApiServiceImplTest {
    private static final String SUBMISSION_ID = "submissionId";

    @InjectMocks
    private StateApiServiceImpl service;

    @Mock
    private AuthorizationCodeRepository acRepo;

    @Mock
    private StateProfileRepository spRepo;

    @Mock
    private DirectFileClient dfClient;

    @Mock
    private ExportedFactsClient efClient;

    @Mock
    private AuthorizationTokenService authorizationTokenService;

    @Mock
    private CachedDataService cachedDS;

    @Mock
    private StateApiS3Client stateApiS3Client;

    @Mock
    private ObjectMapper mapper;

    @Mock
    private CertificationOverrideProperties certProperties;

    ObjectMapper objectMapper = new ObjectMapper();

    @BeforeAll
    static void init() {
        // in Mockito, the @InjectMocks annotation can indeed skip the execution of a
        // bean's @PostConstruct method
        Security.addProvider(new BouncyCastleProvider());
    }

    @Test
    public void testCreateAuthorizationCode_Success() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "DC";
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        AuthorizationCode acEntity = new AuthorizationCode();
        TaxReturnStatus status = new TaxReturnStatus("Accepted", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));
        when(acRepo.save(any())).thenReturn(Mono.just(acEntity));

        Mono<UUID> acMono = service.createAuthorizationCode(acq);

        StepVerifier.create(acMono).expectNextMatches(ac -> ac instanceof UUID).verifyComplete();
    }

    @Test
    public void testCreateAuthorizationCode_SubmissionRejected() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FakeState";
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Rejected", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.TRUE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));
        AuthorizationCode acEntity = new AuthorizationCode();
        Mockito.lenient().when(acRepo.save(any())).thenReturn(Mono.just(acEntity));

        Mono<UUID> acMono = service.createAuthorizationCode(acq);

        StepVerifier.create(acMono)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_TAX_RETURN_NOT_ACCEPTED");
                })
                .verify();
    }

    @Test
    public void testCreateAuthorizationCode_TaxNotFiled() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FakeState";
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Error", false);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));

        Mono<UUID> acMono = service.createAuthorizationCode(acq);

        StepVerifier.create(acMono)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_TAX_RETURN_NOT_FOUND");
                })
                .verify();
    }

    @Test
    public void testCreateAuthorizationCode_NotAcceptedOnlyAllowSubmissionPending() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FakeState";
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        AuthorizationCode acEntity = new AuthorizationCode();
        TaxReturnStatus status = new TaxReturnStatus("Pending", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));
        when(acRepo.save(any())).thenReturn(Mono.just(acEntity));

        Mono<UUID> acMono = service.createAuthorizationCode(acq);

        StepVerifier.create(acMono).expectNextMatches(ac -> ac instanceof UUID).verifyComplete();
    }

    @Test
    public void testCreateAuthorizationCode_NotAcceptedOnlyBlockRejected() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FakeState";
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Rejected", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));

        Mono<UUID> acMono = service.createAuthorizationCode(acq);

        StepVerifier.create(acMono)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable)
                            .isInstanceOf(StateApiException.class)
                            .hasMessage("E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING");
                })
                .verify();
    }

    @Test
    public void testAuthorize_Success() throws Exception {
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        ac.setExpiresAt(new Timestamp(System.currentTimeMillis() + 1000));
        ac.setStateCode("FS");
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "FS");

        when(acRepo.getByAuthorizationCode(ac.getAuthorizationCode())).thenReturn(Mono.just(ac));

        Mono<AuthorizationCode> entityMono = service.authorize(saCode);

        StepVerifier.create(entityMono)
                .expectNextMatches(entity -> entity.getTaxReturnUuid().equals(ac.getTaxReturnUuid()))
                .verifyComplete();
    }

    @Test
    public void testAuthorize_Failure() {
        ClientJwtClaim claim = new ClientJwtClaim();
        String accountId = "the-account-id";
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setStateCode("FS");
        claim.setAccountId(accountId);
        claim.setAuthorizationCode(authorizationCode.toString());
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "FS");

        when(acRepo.getByAuthorizationCode(ac.getAuthorizationCode())).thenReturn(Mono.empty());

        Mono<AuthorizationCode> entityMono = service.authorize(saCode);

        StepVerifier.create(entityMono).expectError(StateApiException.class).verify();
    }

    @Test
    public void testAuthorize_StateCodeMismatch() {
        ClientJwtClaim claim = new ClientJwtClaim();
        String accountId = "the-account-id";
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setStateCode("FS");
        claim.setAccountId(accountId);
        claim.setAuthorizationCode(authorizationCode.toString());
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "DC");

        when(acRepo.getByAuthorizationCode(ac.getAuthorizationCode())).thenReturn(Mono.just(ac));

        Mono<AuthorizationCode> entityMono = service.authorize(saCode);

        StepVerifier.create(entityMono).expectError(StateApiException.class).verify();
    }

    // Tests for authorization token
    @Test
    public void givenValidAuthCodeRequest_whenGeneratingAuthorizationToken_thenSuccessfullyReturnsToken() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "DC";
        // given
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        AuthorizationTokenClaims requestClaims = objectMapper.convertValue(acq, AuthorizationTokenClaims.class);
        TaxReturnStatus status = new TaxReturnStatus("Accepted", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));
        when(authorizationTokenService.generateAndEncrypt(requestClaims))
                .thenReturn(Mono.just("gajoagj.gjsalgj.fgegaj"));
        when(mapper.convertValue(acq, AuthorizationTokenClaims.class)).thenReturn(requestClaims);

        // when
        Mono<String> generateTokenMono = service.generateAuthorizationToken(acq);

        StepVerifier.create(generateTokenMono)
                // then
                .assertNext((token) -> {
                    assertThat(token).isNotBlank();
                })
                .verifyComplete();
    }

    @Test
    public void givenRequestForRejectedSubmission_whenGeneratingAuthorizationToken_thenReturnsError() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FS";
        // given
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Rejected", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.TRUE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));

        // when
        Mono<String> generateTokenMono = service.generateAuthorizationToken(acq);

        StepVerifier.create(generateTokenMono)
                // then
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_TAX_RETURN_NOT_ACCEPTED");
                })
                .verify();
    }

    @Test
    public void givenRequestForUnknownTaxReturn_whenGeneratingAuthorizationToken_thenReturnsError() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FS";
        // given
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Error", false);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));

        // when
        Mono<String> generateTokenMono = service.generateAuthorizationToken(acq);

        StepVerifier.create(generateTokenMono)
                // then
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_TAX_RETURN_NOT_FOUND");
                })
                .verify();
    }

    @Test
    public void
            givenAuthCodeRequestForStateThatAllowsPending_whenGeneratingAuthorizationToken_thenSuccessfullyReturnsToken() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FS";
        // given
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        AuthorizationTokenClaims requestClaims = objectMapper.convertValue(acq, AuthorizationTokenClaims.class);
        TaxReturnStatus status = new TaxReturnStatus("Pending", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));
        when(authorizationTokenService.generateAndEncrypt(requestClaims))
                .thenReturn(Mono.just("gajoagj.gjsalgj.fgegaj"));
        when(mapper.convertValue(acq, AuthorizationTokenClaims.class)).thenReturn(requestClaims);

        // when
        Mono<String> generateTokenMono = service.generateAuthorizationToken(acq);

        StepVerifier.create(generateTokenMono)
                // then
                .assertNext((token) -> {
                    assertThat(token).isNotBlank();
                })
                .verifyComplete();
    }

    @Test
    public void givenAuthCodeRequestForStateThatOnlyAllowsAccepted_whenGeneratingAuthorizationToken_thenReturnsError() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FS";
        // given
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Rejected", true);

        StateProfileDTO sp = new StateProfileDTO(createStateProfile(stateCode, Boolean.FALSE));

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(sp));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));

        // when
        Mono<String> generateTokenMono = service.generateAuthorizationToken(acq);

        StepVerifier.create(generateTokenMono)
                // then
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable)
                            .isInstanceOf(StateApiException.class)
                            .hasMessage("E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING");
                })
                .verify();
    }

    @Test
    public void givenGeneratingAuthorizationToken_whenInternalException_thenReturnsError() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String stateCode = "FS";
        // given/when
        AuthCodeRequest acq = new AuthCodeRequest(taxReturnUuid, "tin", taxYear, stateCode, "submissionId");
        TaxReturnStatus status = new TaxReturnStatus("Pending", true);

        when(cachedDS.getStateProfileByStateCode(stateCode))
                .thenReturn(Mono.error(new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR)));
        when(dfClient.getStatus(taxYear, taxReturnUuid, SUBMISSION_ID)).thenReturn(Mono.just(status));

        Mono<String> generateTokenMono = service.generateAuthorizationToken(acq);

        StepVerifier.create(generateTokenMono)
                // then
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_INTERNAL_SERVER_ERROR");
                })
                .verify();
    }

    @Test
    public void givenTaxReturnStatusIndicatesTheXmlExists_whenRetrieveTaxReturnXml_thenReturnsTaxReturnXml() {
        String xmlData = "<xml>data</xml>";
        String submissionId = "the-submission-id";
        String status = "accepted";
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;

        TaxReturnStatus taxReturnStatus = new TaxReturnStatus(status, true);

        when(dfClient.getStatus(taxYear, taxReturnUuid, submissionId)).thenReturn(Mono.just(taxReturnStatus));
        when(stateApiS3Client.getTaxReturnXml(taxYear, taxReturnUuid, submissionId))
                .thenReturn(Mono.just(xmlData));

        Mono<TaxReturnXml> trMono = service.retrieveTaxReturnXml(taxYear, taxReturnUuid, submissionId);

        StepVerifier.create(trMono)
                .expectNextMatches(tr -> tr.xml().equals(xmlData) && tr.status().equals(status))
                .verifyComplete();
    }

    @Test
    public void givenGetStatusReturnsError_whenRetrieveTaxReturnXml_thenReturnsError() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String submissionId = "the-submission-id";

        when(dfClient.getStatus(taxYear, taxReturnUuid, submissionId)).thenReturn(Mono.error(new RuntimeException()));

        Mono<TaxReturnXml> trMono = service.retrieveTaxReturnXml(taxYear, taxReturnUuid, submissionId);

        StepVerifier.create(trMono).expectError(RuntimeException.class).verify();
    }

    @Test
    public void givenGetTaxReturnXmlReturnsError_whenRetrieveTaxReturnXml_thenReturnsError() {
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;
        String submissionId = "the-submission-id";

        TaxReturnStatus taxReturnStatus = new TaxReturnStatus("status", true);
        when(dfClient.getStatus(taxYear, taxReturnUuid, submissionId)).thenReturn(Mono.just(taxReturnStatus));

        when(stateApiS3Client.getTaxReturnXml(taxYear, taxReturnUuid, submissionId))
                .thenReturn(Mono.error(new RuntimeException()));

        Mono<TaxReturnXml> trMono = service.retrieveTaxReturnXml(taxYear, taxReturnUuid, submissionId);

        StepVerifier.create(trMono).expectError(RuntimeException.class).verify();
    }

    @Test
    public void givenStatusIsRejected_whenRetrieveTaxReturnXml_thenReturnsTaxReturnNotAcceptedOrPendingError() {
        String submissionId = "the-submission-id";
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;

        TaxReturnStatus taxReturnStatus = new TaxReturnStatus(DirectFileClient.STATUS_REJECTED, false);
        when(dfClient.getStatus(taxYear, taxReturnUuid, submissionId)).thenReturn(Mono.just(taxReturnStatus));

        Mono<TaxReturnXml> trMono = service.retrieveTaxReturnXml(taxYear, taxReturnUuid, submissionId);

        StepVerifier.create(trMono)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable)
                            .isInstanceOf(StateApiException.class)
                            .hasMessage(StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING.toString());
                })
                .verify();
    }

    @Test
    public void givenStatusIsError_whenRetrieveTaxReturnXml_thenReturnsInternalServerError() {
        String submissionId = "the-submission-id";
        UUID taxReturnUuid = UUID.randomUUID();
        int taxYear = 2022;

        TaxReturnStatus taxReturnStatus = new TaxReturnStatus(DirectFileClient.STATUS_ERROR, false);
        when(dfClient.getStatus(taxYear, taxReturnUuid, submissionId)).thenReturn(Mono.just(taxReturnStatus));

        Mono<TaxReturnXml> trMono = service.retrieveTaxReturnXml(taxYear, taxReturnUuid, submissionId);

        StepVerifier.create(trMono)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable)
                            .isInstanceOf(StateApiException.class)
                            .hasMessage(StateApiErrorCode.E_INTERNAL_SERVER_ERROR.toString());
                })
                .verify();
    }

    @Test
    public void testEncryptTaxReturnXml_Success() throws Exception {
        String xmlData = "<?xml version='1.0' ?><taxreturn>blahblah</taxreturn>";

        var exportedFacts = new HashMap<String, Object>();
        exportedFacts.put("filers", List.of(Map.of("firstName", "Tom", "lastName", "Smith")));

        TaxReturnToExport trExport = new TaxReturnToExport("accepted", "the-submission-id", xmlData, exportedFacts);
        String accountId = "fakestate";
        StateProfile sp = new StateProfile();
        sp.setCertLocation("src/test/resources/certificates/fakestate.cer");
        sp.setArchived(false);

        when(cachedDS.getStateProfile(accountId)).thenReturn(Mono.just(sp));

        InputStream is = new FileInputStream(sp.getCertLocation());
        X509Certificate cert;

        CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
        cert = (X509Certificate) certFactory.generateCertificate(is);

        when(cachedDS.retrievePublicKeyFromCert(any(), any())).thenReturn(Mono.just(cert.getPublicKey()));

        Mono<EncryptData> edMono = service.encryptTaxReturn(trExport, "fakestate");

        StepVerifier.create(edMono)
                .expectNextMatches(ed -> {
                    try {
                        // decrypt the session secret with private key
                        String privateKeyPath = "src/test/resources/certificates/fakestate.key";
                        byte[] secret = rsaDecryptWithPrivateKey(
                                Base64.getDecoder().decode(ed.encodedSecret()), privateKeyPath);
                        // decrypt the data using the secret and iv
                        byte[] decryptedString = aesGcmDecrypt(
                                Base64.getDecoder().decode(ed.encodedAndEncryptedData()),
                                secret,
                                Base64.getDecoder().decode(ed.encodedIV()),
                                Base64.getDecoder().decode(ed.encodedAuthenticationTag()));
                        // compare decrypted data with original xml
                        return new String(decryptedString).equals(new Gson().toJson(trExport));
                    } catch (Exception e) {
                        return false;
                    }
                })
                .verifyComplete();
    }

    @Test
    public void testEncryptTaxReturnXml_NullSubmission() throws Exception {
        String xmlData = "<?xml version='1.0' ?><taxreturn>blahblah</taxreturn>";

        var exportedFacts = new HashMap<String, Object>();
        exportedFacts.put("filers", List.of(Map.of("firstName", "Tom", "lastName", "Smith")));

        TaxReturnToExport trExport = new TaxReturnToExport("accepted", "the-submission-id", xmlData, exportedFacts);
        String accountId = "fakestate";
        StateProfile sp = new StateProfile();
        sp.setCertLocation("src/test/resources/certificates/fakestate.cer");
        sp.setArchived(false);

        when(cachedDS.getStateProfile(accountId)).thenReturn(Mono.just(sp));

        InputStream is = new FileInputStream(sp.getCertLocation());
        X509Certificate cert;

        CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
        cert = (X509Certificate) certFactory.generateCertificate(is);

        when(cachedDS.retrievePublicKeyFromCert(any(), any())).thenReturn(Mono.just(cert.getPublicKey()));

        Mono<EncryptData> edMono = service.encryptTaxReturn(trExport, "fakestate");

        StepVerifier.create(edMono)
                .expectNextMatches(ed -> {
                    try {
                        // decrypt the session secret with private key
                        String privateKeyPath = "src/test/resources/certificates/fakestate.key";
                        byte[] secret = rsaDecryptWithPrivateKey(
                                Base64.getDecoder().decode(ed.encodedSecret()), privateKeyPath);
                        // decrypt the data using the secret and iv
                        byte[] decryptedString = aesGcmDecrypt(
                                Base64.getDecoder().decode(ed.encodedAndEncryptedData()),
                                secret,
                                Base64.getDecoder().decode(ed.encodedIV()),
                                Base64.getDecoder().decode(ed.encodedAuthenticationTag()));
                        // compare decrypted data with original xml
                        return new String(decryptedString).equals(new Gson().toJson(trExport));
                    } catch (Exception e) {
                        return false;
                    }
                })
                .verifyComplete();
    }

    @Test
    public void testLookupStateProfile() {
        // given
        String stateCode = "DC";
        var sp = createStateProfile(stateCode, Boolean.FALSE);

        List<StateRedirect> stateRedirects = new ArrayList<>();
        var stateRedirect1 = mock(StateRedirect.class);
        when(stateRedirect1.getRedirectUrl()).thenReturn("http://redirect.state.system/1");
        var stateRedirect2 = mock(StateRedirect.class);
        when(stateRedirect2.getRedirectUrl()).thenReturn("http://redirect.state.system/2");

        stateRedirects.add(stateRedirect1);
        stateRedirects.add(stateRedirect2);

        List<StateLanguage> stateLanguages = new ArrayList<>();
        var stateLanguage1 = mock(StateLanguage.class);
        when(stateLanguage1.getDfLanguageCode()).thenReturn("en");
        when(stateLanguage1.getStateLanguageCode()).thenReturn("english");

        var stateLanguage2 = mock(StateLanguage.class);
        when(stateLanguage2.getDfLanguageCode()).thenReturn("es");
        when(stateLanguage2.getStateLanguageCode()).thenReturn("spanish");

        stateLanguages.add(stateLanguage1);
        stateLanguages.add(stateLanguage2);

        var spDto = new StateProfileDTO(sp, stateRedirects, stateLanguages);

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(spDto));

        // when
        var retrievedStateProfile = service.lookupStateProfile(stateCode);

        // then
        StepVerifier.create(retrievedStateProfile)
                .expectNextMatches(s -> s.redirectUrls()
                                .containsAll(List.of(stateRedirect1.getRedirectUrl(), stateRedirect2.getRedirectUrl()))
                        && "english".equals(s.languages().get("en"))
                        && "spanish".equals(s.languages().get("es"))
                        && "url".equals(s.departmentOfRevenueUrl())
                        && "url".equals(s.filingRequirementsUrl()))
                .expectComplete()
                .verify();
    }

    @Test
    public void testLookupStateProfile_Archived() {
        // given
        String stateCode = "DC";
        var sp = createStateProfile(stateCode, Boolean.FALSE);
        sp.setArchived(true);

        List<StateRedirect> stateRedirects = new ArrayList<>();
        var stateRedirect1 = mock(StateRedirect.class);
        when(stateRedirect1.getRedirectUrl()).thenReturn("http://redirect.state.system/1");
        var stateRedirect2 = mock(StateRedirect.class);
        when(stateRedirect2.getRedirectUrl()).thenReturn("http://redirect.state.system/2");

        stateRedirects.add(stateRedirect1);
        stateRedirects.add(stateRedirect2);

        var spDto = new StateProfileDTO(sp, stateRedirects, new ArrayList<>());

        when(cachedDS.getStateProfileByStateCode(stateCode)).thenReturn(Mono.just(spDto));

        // when
        var retrievedStateProfile = service.lookupStateProfile(stateCode);

        // then
        StepVerifier.create(retrievedStateProfile)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_ACCOUNT_ARCHIVED");
                })
                .verify();
    }

    @Test
    public void testLookupStateProfile_NotExist() {

        String stateCode = "AR";
        when(cachedDS.getStateProfileByStateCode(stateCode))
                .thenReturn(Mono.error(new StateNotExistException(StateApiErrorCode.E_STATE_NOT_EXIST) {}));

        // when
        var retrievedStateProfile = service.lookupStateProfile(stateCode);

        // then
        StepVerifier.create(retrievedStateProfile)
                .expectErrorMatches(e ->
                        e instanceof StateNotExistException && e.getMessage().contains("E_STATE_NOT_EXIST"))
                .verify();
    }

    @Test
    public void testLookupStateProfile_Exception() {

        String stateCode = "DC";
        when(cachedDS.getStateProfileByStateCode(stateCode))
                .thenReturn(Mono.error(new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR) {}));

        // when
        var retrievedStateProfile = service.lookupStateProfile(stateCode);

        // then
        StepVerifier.create(retrievedStateProfile)
                .expectErrorMatches(
                        e -> e instanceof StateApiException && e.getMessage().equals("E_INTERNAL_SERVER_ERROR"))
                .verify();
    }

    @Test
    public void testGetStateProfile_Archived() {
        // given
        String accountId = "the-account-id";
        StateProfile sp = new StateProfile();
        sp.setArchived(true);

        when(cachedDS.getStateProfile(accountId)).thenReturn(Mono.just(sp));

        // when
        var retrievedStateProfile = service.getStateProfile(accountId);

        // then
        StepVerifier.create(retrievedStateProfile)
                .expectErrorSatisfies(throwable -> {
                    assertThat(throwable).isInstanceOf(StateApiException.class).hasMessage("E_ACCOUNT_ARCHIVED");
                })
                .verify();
    }

    @Test
    public void testRetrieveExportedFacts_Success() {
        String submissionId = "the-submission-id";
        String accountId = "fakestate";

        var exportedFacts = new HashMap<String, Object>();
        exportedFacts.put("filers", List.of(Map.of("firstName", "Tom", "lastName", "Smith")));
        var getStateExportedFactsResponse = new GetStateExportedFactsResponse(exportedFacts);

        ReflectionTestUtils.setField(service, "exported_facts_enabled", true);
        when(efClient.getExportedFacts(submissionId, "NY", accountId))
                .thenReturn(Mono.just(getStateExportedFactsResponse));
        var efMono = service.retrieveExportedFacts(submissionId, "NY", accountId);

        StepVerifier.create(efMono)
                .expectNextMatches(ef -> ef.equals(exportedFacts))
                .verifyComplete();
    }

    @Test
    public void testRetrieveExportedFacts_enabled_false() {
        String submissionId = "the-submission-id";
        String accountId = "fakestate";

        ReflectionTestUtils.setField(service, "exported_facts_enabled", false);
        var efMono = service.retrieveExportedFacts(submissionId, "NY", accountId);

        StepVerifier.create(efMono).expectNextMatches(Map::isEmpty).verifyComplete();
    }

    @Test
    public void testRetrieveExportedFacts_WithException() {
        String submissionId = "the-submission-id";
        String accountId = "fakestate";

        ReflectionTestUtils.setField(service, "exported_facts_enabled", true);
        when(efClient.getExportedFacts(submissionId, "NY", accountId)).thenReturn(Mono.error(new RuntimeException()));
        var efMono = service.retrieveExportedFacts(submissionId, "NY", accountId);

        StepVerifier.create(efMono).expectError(RuntimeException.class).verify();
    }

    private StateProfile createStateProfile(String stateCode, Boolean flag) {

        StateProfile sp = new StateProfile();
        sp.setAccountId("123456");
        sp.setStateCode(stateCode); //
        sp.setCertLocation("x/y");
        sp.setAcceptedOnly(flag);
        sp.setArchived(false);
        sp.setDepartmentOfRevenueUrl("url");
        sp.setFilingRequirementsUrl("url");
        return sp;
    }
}
