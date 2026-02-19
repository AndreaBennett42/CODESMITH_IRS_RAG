package gov.irs.directfile.stateapi.configuration;

import java.net.URI;

import lombok.AllArgsConstructor;
import org.apache.commons.lang3.StringUtils;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.S3AsyncClientBuilder;
import software.amazon.awssdk.services.sts.StsClient;
import software.amazon.awssdk.services.sts.auth.StsAssumeRoleCredentialsProvider;
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

@Configuration
@AllArgsConstructor
public class S3ClientConfiguration {
    private final EncryptionClientConfiguration encryptionClientConfiguration;
    private final S3ConfigurationProperties s3ConfigurationProperties;
    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    @Primary
    S3AsyncClient getS3AsyncClient() {
        return baseS3Client(s3ConfigurationProperties);
    }

    @Bean
    @Profile("aws")
    public S3AsyncEncryptionClient s3AsyncEncryptionClient() {
        return S3AsyncEncryptionClient.builder()
                .wrappedClient(baseS3Client(s3ConfigurationProperties))
                .cryptoMaterialsManager(
                        encryptionClientConfiguration.kmsCrypto(encryptionClientConfiguration.getS3KmsWrappingKeyArn()))
                .build();
    }

    @Bean
    @Profile("!aws")
    public S3AsyncEncryptionClient localS3AsyncEncryptionClient() {
        return S3AsyncEncryptionClient.builder()
                .wrappedClient(baseS3Client(s3ConfigurationProperties))
                .aesKey(encryptionClientConfiguration.getLocalAesWrappingKey())
                .build();
    }

    private S3AsyncClient baseS3Client(S3ConfigurationProperties s3ConfigurationProperties) {
        S3AsyncClientBuilder builder = S3AsyncClient.builder().region(Region.of(s3ConfigurationProperties.getRegion()));

        if (s3ConfigurationProperties.getAssumeRoleArn() != null
                && StringUtils.isNotBlank(s3ConfigurationProperties.getAssumeRoleArn())) {
            builder.credentialsProvider(StsAssumeRoleCredentialsProvider.builder()
                    .stsClient(StsClient.builder()
                            .region(Region.of(s3ConfigurationProperties.getRegion()))
                            .credentialsProvider(awsCredentialsProvider)
                            .build())
                    .refreshRequest(AssumeRoleRequest.builder()
                            .roleArn(s3ConfigurationProperties.getAssumeRoleArn())
                            .roleSessionName(s3ConfigurationProperties.getAssumeRoleSessionName())
                            .durationSeconds(Integer.parseInt(s3ConfigurationProperties.getAssumeRoleDurationSeconds()))
                            .build())
                    .build());
        } else {
            builder.credentialsProvider(StaticCredentialsProvider.create(AwsBasicCredentials.create(
                    s3ConfigurationProperties.getAccessKey(), s3ConfigurationProperties.getSecretKey())));
        }

        if (s3ConfigurationProperties.getEndPoint() != null
                && StringUtils.isNotBlank(s3ConfigurationProperties.getEndPoint())) {
            builder.endpointOverride(URI.create(s3ConfigurationProperties.getEndPoint()))
                    .forcePathStyle(true); // <-- this fixes the UnknonwnHost issue with localstack
        }

        return builder.build();
    }
}
