package gov.irs.directfile.stateapi.encryption;

import java.io.FileNotFoundException;
import java.security.PublicKey;
import java.security.cert.CertificateException;
import java.security.interfaces.RSAPublicKey;
import java.util.Base64;
import java.util.UUID;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import io.micrometer.common.util.StringUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtException;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.model.ClientJwtClaim;

@Slf4j
@SuppressWarnings({"PMD.AvoidUncheckedExceptionsInSignatures", "PMD.PreserveStackTrace"})
public class JwtVerifier {

    static ObjectMapper objectMapper = new ObjectMapper();

    static {
        // Jwt treats 'iat' as Instant type, cannot convert claim map to Java Bean via convertValue without setting
        // JavaTimeModule
        JavaTimeModule module = new JavaTimeModule();
        objectMapper.registerModule(module);
    }

    public static ClientJwtClaim verifyJwt(String jwtString, PublicKey publicKey)
            throws CertificateException, FileNotFoundException, JwtException, JsonMappingException,
                    JsonProcessingException {

        // assume publicKey is RSAPublicKey, if not, that's okay, let it throw error
        JwtDecoder jwtDecoder =
                NimbusJwtDecoder.withPublicKey((RSAPublicKey) publicKey).build();
        Jwt jwt = jwtDecoder.decode(jwtString);

        return objectMapper.convertValue(jwt.getClaims(), ClientJwtClaim.class);
    }

    // Note: use this method to do payload simple verification.
    public static String getAccountId(String jwtToken) {
        ClientJwtClaim claim;
        try {
            String[] parts = jwtToken.split("\\.");
            // Note: We need to use java.util.Base64's getUrlDecoder, instead of org.bouncycastle.util.encoders.Base64.
            // b/c here the encoded string is coming from http request.
            byte[] payload = Base64.getUrlDecoder().decode(parts[1]);
            ObjectMapper objectMapper = new ObjectMapper();
            claim = objectMapper.readValue(payload, ClientJwtClaim.class);
        } catch (Exception e) {
            log.error(
                    "getAccountId() failed, could not extract account id from JWT Bearer token, {}, error: {}",
                    e.getClass(),
                    e.getMessage());
            throw new StateApiException(StateApiErrorCode.E_BAD_JWT_BEARER_TOKEN);
        }

        String accountId = claim.getAccountId();
        if (StringUtils.isBlank(accountId)) {
            log.error("Account id is missing from JWT Bearer token");
            throw new StateApiException(StateApiErrorCode.E_ACCOUNT_ID_MISSING_IN_JWT_TOKEN);
        } else {
            try {
                UUID.fromString(claim.getAuthorizationCode());
            } catch (IllegalArgumentException e) {
                log.error("Authorization code is not valid: {}", claim.getAuthorizationCode());
                throw new StateApiException(StateApiErrorCode.E_AUTHORIZATION_CODE_INVALID_FORMAT);
            }
        }

        return accountId;
    }
}
