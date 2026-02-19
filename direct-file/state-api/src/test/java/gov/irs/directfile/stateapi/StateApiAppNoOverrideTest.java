package gov.irs.directfile.stateapi;

import java.security.Security;
import java.util.*;

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
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.web.client.RestTemplate;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

import gov.irs.directfile.stateapi.configuration.S3ConfigurationProperties;
import gov.irs.directfile.stateapi.encryption.JwtSigner;
import gov.irs.directfile.stateapi.model.ExportResponse;
import gov.irs.directfile.stateapi.repository.AuthorizationCodeRepository;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT,
        properties = {"server.port=8082"})
@ActiveProfiles({"development", "test", "integration-test"})
@Slf4j
@SuppressWarnings("null")
@EnabledIfSystemProperty(named = "runIntegrationTests", matches = "true")
public class StateApiAppNoOverrideTest {

    private static final String PRIVATE_KEY_PATH = "src/test/resources/certificates/fakestate.key";
    private static final String URL_EXPORT = "http://localhost:8082/state-api/export-return";
    private final RestTemplate restTemplate = new RestTemplate();
    private final HttpHeaders headers = new HttpHeaders();
    private HttpEntity<Map<String, Object>> requestEntity;

    @Autowired
    AuthorizationCodeRepository authorizationCodeRepository;

    @Autowired
    S3AsyncEncryptionClient s3AsyncEncryptionClient;

    @Autowired
    S3ConfigurationProperties s3ConfigurationProperties;

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        log.info("Set cert-location-override=false");
        registry.add("direct-file.cert-location-override", () -> "");
    }

    @BeforeEach
    public void setUp() {
        Security.addProvider(new BouncyCastleProvider());
        headers.setContentType(MediaType.APPLICATION_JSON);
    }

    @Test
    public void integrationTests_CertificateExpired() throws Exception {

        String ac = "d6e34be4-11df-4b5e-808f-bc48a9f4870b";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"112233\",\"sub\":\"" + ac + "\",\"iat\":1516239022}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);
        // Set the desired header(s)
        headers.set("account-id", "112233");
        headers.set("Authorization", "Bearer " + jwtToken);

        requestEntity = new HttpEntity<>(headers);
        var response = restTemplate.exchange(URL_EXPORT, HttpMethod.GET, requestEntity, ExportResponse.class);

        ExportResponse responseBody = response.getBody();

        assertNotNull(responseBody);
        assertEquals("error", responseBody.getStatus());
        assertEquals("E_CERTIFICATE_EXPIRED", responseBody.getError());

        log.info("Test succeeded with expected error: E_CERTIFICATE_EXPIRED");
    }

    @Test
    public void integrationTests_InternalServerError() throws Exception {

        String ac = "d6e34be4-11df-4b5e-808f-bc48a9f4870b";
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"112234\",\"sub\":\"" + ac + "\",\"iat\":1516239022}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, PRIVATE_KEY_PATH);
        // Set the desired header(s)
        headers.set("account-id", "112234");
        headers.set("Authorization", "Bearer " + jwtToken);

        requestEntity = new HttpEntity<>(headers);
        var response = restTemplate.exchange(URL_EXPORT, HttpMethod.GET, requestEntity, ExportResponse.class);

        ExportResponse responseBody = response.getBody();

        assertNotNull(responseBody);
        assertEquals("error", responseBody.getStatus());
        assertEquals("E_INTERNAL_SERVER_ERROR", responseBody.getError());

        log.info("Test succeeded with expected error: E_INTERNAL_SERVER_ERROR");
    }
}
