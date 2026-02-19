package gov.irs.directfile.status.config;

import java.net.URI;
import java.util.Base64;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

import lombok.AllArgsConstructor;
import lombok.NonNull;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.kms.KmsClient;
import software.amazon.encryption.s3.materials.CryptographicMaterialsManager;
import software.amazon.encryption.s3.materials.DefaultCryptoMaterialsManager;
import software.amazon.encryption.s3.materials.KmsKeyring;

@Configuration
@AllArgsConstructor
@EnableConfigurationProperties(EncryptionConfiguration.class)
public class EncryptionClientConfiguration {
    private final EncryptionConfiguration encryptionConfig;
    private final AwsCredentialsProvider awsCredentialsProvider;

    public CryptographicMaterialsManager kmsCrypto(@NonNull String kmsWrappingKeyArn) {
        return DefaultCryptoMaterialsManager.builder()
                .keyring(KmsKeyring.builder()
                        .kmsClient(regionalKmsClient())
                        .wrappingKeyId(kmsWrappingKeyArn)
                        .build())
                .build();
    }

    public SecretKey getLocalAesWrappingKey() {
        return new SecretKeySpec(Base64.getDecoder().decode(encryptionConfig.getLocalWrappingKey()), "AES");
    }

    protected KmsClient regionalKmsClient() {
        return KmsClient.builder()
                .region(Region.of(encryptionConfig.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(encryptionConfig.getKmsEndpoint()))
                .build();
    }
}
