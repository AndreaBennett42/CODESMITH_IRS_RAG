package gov.irs.directfile.api.authorization.config;

import java.net.URI;

import lombok.AllArgsConstructor;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.sts.StsClient;
import software.amazon.awssdk.services.sts.auth.StsAssumeRoleCredentialsProvider;
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest;

import gov.irs.directfile.api.config.S3ConfigurationProperties;

@Configuration
@EnableConfigurationProperties(S3ConfigurationProperties.class)
@AllArgsConstructor
public class StorageConfiguration {

    private final AwsCredentialsProvider awsCredentialsProvider;

    @Profile("aws")
    @Bean("s3WithoutEncryption")
    S3Client s3Client(S3ConfigurationProperties s3ConfigurationProperties) {
        return S3Client.builder()
                .region(Region.of(s3ConfigurationProperties.getRegion()))
                .credentialsProvider(StsAssumeRoleCredentialsProvider.builder()
                        .stsClient(StsClient.builder()
                                .region(Region.of(s3ConfigurationProperties.getRegion()))
                                .credentialsProvider(awsCredentialsProvider)
                                .build())
                        .refreshRequest(AssumeRoleRequest.builder()
                                .roleArn(s3ConfigurationProperties.getS3().getAssumeRoleArn())
                                .roleSessionName(
                                        s3ConfigurationProperties.getS3().getAssumeRoleSessionName())
                                .durationSeconds(
                                        s3ConfigurationProperties.getS3().getAssumeRoleDurationSeconds())
                                .build())
                        .build())
                .endpointOverride(URI.create(s3ConfigurationProperties.getS3().getEndpoint()))
                .build();
    }

    @Profile("!aws")
    @Bean("s3WithoutEncryption")
    S3Client localS3Client(S3ConfigurationProperties s3ConfigurationProperties) {
        return S3Client.builder()
                .region(Region.of(s3ConfigurationProperties.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(s3ConfigurationProperties.getS3().getEndpoint()))
                .build();
    }
}
