package gov.irs.directfile.submit.config;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import software.amazon.awssdk.services.kms.KmsClient;
import software.amazon.encryption.s3.materials.CryptographicMaterialsManager;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(classes = {EncryptionClientConfig.class, AWSCredentialsConfig.class})
class EncryptionClientConfigTest {

    @Autowired
    EncryptionClientConfig encryptionClientConfig;

    @Test
    void regionalKmsClientIsNotNull() {
        KmsClient kmsClient = encryptionClientConfig.regionalKmsClient();
        assertNotNull(kmsClient);
    }

    @Test
    void kmsCryptoIsNotNull() {
        String testKmsKeyArn = "test-kms-arn";
        CryptographicMaterialsManager cryptographicMaterialsManager = encryptionClientConfig.kmsCrypto(testKmsKeyArn);
        assertNotNull(cryptographicMaterialsManager);
    }
}
