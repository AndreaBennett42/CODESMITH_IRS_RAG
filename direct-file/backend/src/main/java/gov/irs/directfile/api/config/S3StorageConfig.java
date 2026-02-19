package gov.irs.directfile.api.config;

import java.net.URI;
import java.util.Base64;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.encryption.s3.S3EncryptionClient;

@Slf4j
@Configuration
@AllArgsConstructor
public class S3StorageConfig {

    private final AwsCredentialsProvider awsCredentialsProvider;

    @Profile("!aws")
    @EnableConfigurationProperties({S3ConfigurationProperties.class, LocalEncryptionConfigurationProperties.class})
    class Local {
        @Bean
        @Primary
        S3EncryptionClient s3LocalEncryptionClient(
                S3ConfigurationProperties s3ConfigurationProperties,
                LocalEncryptionConfigurationProperties localEncryptionConfigurationProperties) {
            log.warn("S3: Using local encryption without AWS KMS. Not appropriate for deployed environments!");
            return S3EncryptionClient.builder()
                    .wrappedClient(staticCredentialClient(s3ConfigurationProperties))
                    .wrappedAsyncClient(asyncStaticCredentialClient(s3ConfigurationProperties))
                    .aesKey(getLocalAesWrappingKey(localEncryptionConfigurationProperties))
                    .build();
        }

        private S3Client staticCredentialClient(S3ConfigurationProperties s3ConfigurationProperties) {
            return S3Client.builder()
                    .region(Region.of(s3ConfigurationProperties.getRegion()))
                    .credentialsProvider(awsCredentialsProvider)
                    .endpointOverride(
                            URI.create(s3ConfigurationProperties.getS3().getEndpoint()))
                    .build();
        }

        private S3AsyncClient asyncStaticCredentialClient(S3ConfigurationProperties s3ConfigurationProperties) {
            return S3AsyncClient.builder()
                    .region(Region.of(s3ConfigurationProperties.getRegion()))
                    .credentialsProvider(awsCredentialsProvider)
                    .endpointOverride(
                            URI.create(s3ConfigurationProperties.getS3().getEndpoint()))
                    .build();
        }

        private SecretKey getLocalAesWrappingKey(
                LocalEncryptionConfigurationProperties localEncryptionConfigurationProperties) {
            return new SecretKeySpec(
                    Base64.getDecoder().decode(localEncryptionConfigurationProperties.getLocalWrappingKey()), "AES");
        }
    }
}
