package gov.irs.directfile.stateapi.authorization;

import java.io.IOException;
import java.text.ParseException;
import java.util.Base64;
import java.util.UUID;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbusds.jose.JOSEException;
import com.nimbusds.jose.JWSVerifier;
import com.nimbusds.jose.crypto.MACVerifier;
import com.nimbusds.jwt.JWTClaimsSet;
import com.nimbusds.jwt.SignedJWT;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import reactor.test.StepVerifier;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;
import gov.irs.directfile.stateapi.model.AuthCodeRequest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
@ActiveProfiles({"test", "token-integration-test"})
public class AuthorizationTokenServiceIntegrationTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Autowired
    AuthorizationTokenService authorizationTokenService;

    @Autowired
    DataEncryptDecrypt dataEncryptDecrypt;

    @Test
    public void givenTokenWithEncryptedTaxReturnClaims_whenClaimsDecrypted_thenClaimsMatchOriginalRequest() {
        // given
        AuthCodeRequest authCodeRequest =
                new AuthCodeRequest(UUID.randomUUID(), "123-00-4567", 2023, "MA", "123456789AB");
        AuthorizationTokenClaims claimsMap = mapper.convertValue(authCodeRequest, AuthorizationTokenClaims.class);
        StepVerifier.create(authorizationTokenService.generateAndEncrypt(claimsMap))
                .assertNext((token) -> {
                    try {
                        // when
                        byte[] ciphertext = Base64.getUrlDecoder().decode(token);
                        byte[] decrypted = dataEncryptDecrypt.decrypt(ciphertext);
                        SignedJWSParts signedJWSParts = mapper.readValue(decrypted, SignedJWSParts.class);
                        String signedJWS =
                                String.join(".", signedJWSParts.s1(), signedJWSParts.s2(), signedJWSParts.s3());

                        SignedJWT signedJWT = SignedJWT.parse(signedJWS);
                        JWTClaimsSet jwtClaimsSet = signedJWT.getJWTClaimsSet();
                        AuthorizationTokenClaims decryptedClaims = mapper.convertValue(
                                jwtClaimsSet.toJSONObject().get("tax-return-export-metadata"),
                                AuthorizationTokenClaims.class);

                        // then
                        assertEquals(authCodeRequest.getTaxReturnUuid().toString(), decryptedClaims.taxReturnUuid());
                        assertEquals(authCodeRequest.getTaxYear(), decryptedClaims.taxYear());
                        assertEquals(authCodeRequest.getStateCode(), decryptedClaims.stateCode());
                        assertEquals(authCodeRequest.getSubmissionId(), decryptedClaims.submissionId());
                    } catch (ParseException | IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .expectComplete()
                .verify();
    }

    @Test
    public void givenASignedToken_whenVerifyingSignature_thenTokenSignatureIsValid() {
        // given
        AuthCodeRequest authCodeRequest =
                new AuthCodeRequest(UUID.randomUUID(), "123-00-4567", 2023, "MA", "123456789AB");
        AuthorizationTokenClaims claimsMap = mapper.convertValue(authCodeRequest, AuthorizationTokenClaims.class);
        StepVerifier.create(authorizationTokenService.generateAndEncrypt(claimsMap))
                .assertNext((token) -> {
                    try {
                        // when
                        JWSVerifier signatureVerifier = new MACVerifier("GTc+SlI7C7ECPHAhAvIWqn2yAvzAGMVj");
                        byte[] ciphertext = Base64.getUrlDecoder().decode(token);
                        byte[] decrypted = dataEncryptDecrypt.decrypt(ciphertext);
                        SignedJWSParts signedJWSParts = mapper.readValue(decrypted, SignedJWSParts.class);
                        String signedJWS =
                                String.join(".", signedJWSParts.s1(), signedJWSParts.s2(), signedJWSParts.s3());

                        SignedJWT signedJWT = SignedJWT.parse(signedJWS);

                        // then
                        assertTrue(signedJWT.verify(signatureVerifier));
                    } catch (ParseException | JOSEException | IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .expectComplete()
                .verify();
    }
}
