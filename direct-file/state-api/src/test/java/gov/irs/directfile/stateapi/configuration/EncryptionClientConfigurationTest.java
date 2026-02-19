package gov.irs.directfile.stateapi.configuration;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import software.amazon.awssdk.services.kms.KmsClient;
import software.amazon.encryption.s3.materials.CryptographicMaterialsManager;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(classes = {EncryptionClientConfiguration.class, AwsCredentialsConfiguration.class})
@ActiveProfiles("test")
class EncryptionClientConfigurationTest {

    @Autowired
    EncryptionClientConfiguration encryptionClientConfiguration;

    @Test
    void regionalKmsClientIsNotNull() {
        KmsClient kmsClient = encryptionClientConfiguration.regionalKmsClient();
        assertNotNull(kmsClient);
    }

    @Test
    void kmsCryptoIsNotNull() {
        String testKmsKeyArn = "test-kms-arn";
        CryptographicMaterialsManager cryptographicMaterialsManager =
                encryptionClientConfiguration.kmsCrypto(testKmsKeyArn);
        assertNotNull(cryptographicMaterialsManager);
    }
}
