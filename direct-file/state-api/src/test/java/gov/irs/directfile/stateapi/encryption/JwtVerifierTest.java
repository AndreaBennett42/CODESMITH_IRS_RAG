package gov.irs.directfile.stateapi.encryption;

import java.io.FileInputStream;
import java.security.Security;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;

import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.model.ClientJwtClaim;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class JwtVerifierTest {
    @BeforeAll
    static void init() {
        Security.addProvider(new BouncyCastleProvider());
    }

    @Test
    void testCreateAndVerifyJwt() throws Exception {
        // Example JWT header and payload (as JSON strings)
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"1234567890\",\"sub\":\"4638655a-5798-4174-a5a0-37cc3b3cd9a0\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, "src/test/resources/certificates/fakestate.key");

        // Verify the JWT
        CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
        X509Certificate cert = (X509Certificate)
                certFactory.generateCertificate(new FileInputStream("src/test/resources/certificates/fakestate.cer"));
        ClientJwtClaim jwt = JwtVerifier.verifyJwt(jwtToken, cert.getPublicKey());

        assertThat(jwt.getAccountId()).isEqualTo("1234567890");
        assertThat(jwt.getAuthorizationCode()).isEqualTo("4638655a-5798-4174-a5a0-37cc3b3cd9a0");
    }

    @Test
    void testGetAccountId_BadAuthorCode() throws Exception {
        // Example JWT header and payload (as JSON strings)
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"1234567890\",\"sub\":\"a\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, "src/test/resources/certificates/fakestate.key");

        // Use assertThrows to check for an exception
        StateApiException exception = assertThrows(StateApiException.class, () -> JwtVerifier.getAccountId(jwtToken));

        assertThat(exception.getErrorCode()).isEqualTo(StateApiErrorCode.E_AUTHORIZATION_CODE_INVALID_FORMAT);
    }

    @Test
    void testGetAccountId_MissingAccountID() throws Exception {
        // Example JWT header and payload (as JSON strings)
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload = "{\"iss\":\"\",\"sub\":\"a\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, "src/test/resources/certificates/fakestate.key");

        // Use assertThrows to check for an exception
        StateApiException exception = assertThrows(StateApiException.class, () -> JwtVerifier.getAccountId(jwtToken));

        assertThat(exception.getErrorCode()).isEqualTo(StateApiErrorCode.E_ACCOUNT_ID_MISSING_IN_JWT_TOKEN);
    }

    @Test
    void testGetAccountId_BadBearerToken() throws Exception {
        // Create the JWT
        String jwtToken = "Bearer bad-token";

        // Use assertThrows to check for an exception
        StateApiException exception = assertThrows(StateApiException.class, () -> JwtVerifier.getAccountId(jwtToken));

        assertThat(exception.getErrorCode()).isEqualTo(StateApiErrorCode.E_BAD_JWT_BEARER_TOKEN);
    }
}
