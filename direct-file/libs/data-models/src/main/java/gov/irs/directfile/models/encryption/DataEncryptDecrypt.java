package gov.irs.directfile.models.encryption;

import java.util.Map;

import com.amazonaws.encryptionsdk.AwsCrypto;
import com.amazonaws.encryptionsdk.CryptoMaterialsManager;
import com.amazonaws.encryptionsdk.CryptoResult;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.annotation.PostConstruct;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Component
@Slf4j
@AllArgsConstructor
@SuppressFBWarnings(value = "DM_DEFAULT_ENCODING", justification = "Initial Spotbugs Setup")
@SuppressWarnings("PMD.UnusedPrivateMethod")
public class DataEncryptDecrypt {
    private final AwsCrypto awsCrypto;
    private final CryptoMaterialsManager cryptoMaterialsManager;

    public byte[] encrypt(byte[] bytes, Map<String, String> context) {
        CryptoResult<byte[], ?> encryptResult = awsCrypto.encryptData(cryptoMaterialsManager, bytes, context);
        return encryptResult.getResult();
    }

    public byte[] decrypt(byte[] ciphertext) {
        CryptoResult<byte[], ?> decryptResult = awsCrypto.decryptData(cryptoMaterialsManager, ciphertext);
        return decryptResult.getResult();
    }

    @PostConstruct
    private void checkKmsConnection() {
        byte[] testBytes = "something".getBytes();
        try {
            awsCrypto.encryptData(cryptoMaterialsManager, testBytes);
            log.info("encryption setup health check passed");
        } catch (Exception e) {
            log.error("test encrypt operation failed, check configuration");
            throw e;
        }
    }
}
