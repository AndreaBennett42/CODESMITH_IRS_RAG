package gov.irs.directfile.stateapi.model;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Timestamp;
import java.util.UUID;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.bouncycastle.util.encoders.Hex;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.exception.StateApiException;

@Data
@Table(name = "authorization_code")
@Entity
@Slf4j
@SuppressWarnings("PMD.PreserveStackTrace")
public class AuthorizationCode {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    private UUID taxReturnUuid;

    @NotBlank
    private String authorizationCode;

    @NotNull private int taxYear;

    @NotNull private Timestamp expiresAt;

    @NotBlank
    private String stateCode;

    private String submissionId;

    public void setAuthorizationCode(UUID code) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(code.toString().getBytes(StandardCharsets.UTF_8));
            this.authorizationCode = new String(Hex.encode(hash), StandardCharsets.UTF_8);
        } catch (NoSuchAlgorithmException e) {
            log.error("setAuthorizationCode() failed to hash UUID, could not find algorithm");
            throw new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
        }
    }
}
