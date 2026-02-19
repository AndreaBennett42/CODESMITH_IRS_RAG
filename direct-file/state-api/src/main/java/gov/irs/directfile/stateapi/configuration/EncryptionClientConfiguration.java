package gov.irs.directfile.stateapi.configuration;

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
@EnableConfigurationProperties({S3ConfigurationProperties.class, EncryptionConfigurationProperties.class})
public class EncryptionClientConfiguration {
    private final EncryptionConfigurationProperties encryptionConfig;
    private final S3ConfigurationProperties s3ConfigurationProperties;
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

    public String getS3KmsWrappingKeyArn() {
        return encryptionConfig.getS3KmsWrappingKeyArn();
    }

    protected KmsClient regionalKmsClient() {
        return KmsClient.builder()
                .region(Region.of(encryptionConfig.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(encryptionConfig.getKmsEndpoint()))
                .build();
    }
}
