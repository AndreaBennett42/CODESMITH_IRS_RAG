package gov.irs.directfile.stateapi.authorization;

import java.time.Instant;
import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbusds.jose.*;
import com.nimbusds.jose.crypto.MACSigner;
import com.nimbusds.jwt.JWTClaimsSet;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;

@Slf4j
@Service
public class AuthorizationTokenService {
    private final DataEncryptDecrypt dataEncryptDecrypt;
    public static final String EXPORT_CLAIM_KEY = "tax-return-export-metadata";
    private final ObjectMapper mapper = new ObjectMapper();
    private final int authorizationCodeExpiresInterval;
    private final String signingKey;

    public AuthorizationTokenService(
            DataEncryptDecrypt dataEncryptDecrypt,
            @Value("${authorization-token.signing-key}") String signingKey,
            @Value("${authorization-code.expires-interval-seconds: 600}") int authorizationCodeExpiresInterval) {
        this.dataEncryptDecrypt = dataEncryptDecrypt;
        this.signingKey = signingKey;
        this.authorizationCodeExpiresInterval = authorizationCodeExpiresInterval;
    }

    /**
     *  Creates a JWT (json web token) containing tax return export metadata which is first signed and then encrypted.
     */
    public Mono<String> generateAndEncrypt(AuthorizationTokenClaims claims) {
        return Mono.fromCallable(() -> {
            JWSSigner signer = new MACSigner(signingKey);
            Instant issuedAt = Instant.now();
            JWTClaimsSet jwtClaimsSet = new JWTClaimsSet.Builder()
                    .claim(EXPORT_CLAIM_KEY, claims)
                    .issueTime(Date.from(issuedAt))
                    .expirationTime(Date.from(issuedAt.plusSeconds(authorizationCodeExpiresInterval)))
                    .build();
            // create and sign a JWS (json web signature) containing the
            // claims as the payload
            JWSObject jwsObject = new JWSObject(
                    new JWSHeader.Builder(JWSAlgorithm.HS256).build(), new Payload(jwtClaimsSet.toJSONObject()));
            jwsObject.sign(signer);
            // encrypt the token with kms encryption sdk
            return encryptToken(signedJWSToBytes(jwsObject.serialize()));
        });
    }

    private byte[] signedJWSToBytes(String serializedJws) throws JsonProcessingException {
        // break the serialized JWSObject into its period-separated parts
        // before converting to byte array. We must do this prior to encryption to preserve
        // the separate parts of the token (header, signature, and payload)
        String[] serializedJWSParts = serializedJws.split("\\.");
        SignedJWSParts jwsParts =
                new SignedJWSParts(serializedJWSParts[0], serializedJWSParts[1], serializedJWSParts[2]);

        return mapper.writeValueAsBytes(jwsParts);
    }

    private String encryptToken(byte[] claims) {
        Map<String, String> encryptionContext = new HashMap<>();
        encryptionContext.put("system", "DIRECT-FILE");
        encryptionContext.put("type", "STATE-API");

        byte[] ciphertext = dataEncryptDecrypt.encrypt(claims, encryptionContext);
        return Base64.getUrlEncoder().encodeToString(ciphertext);
    }
}
