package gov.irs.directfile.submit.config;

import java.net.URI;
import java.time.Duration;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.sts.StsClient;
import software.amazon.awssdk.services.sts.auth.StsAssumeRoleCredentialsProvider;
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;
import software.amazon.encryption.s3.S3EncryptionClient;

@Slf4j
@Configuration
@AllArgsConstructor
@EnableConfigurationProperties(DocumentStoreConfig.class)
@SuppressWarnings("PMD.AvoidDuplicateLiterals")
public class S3StorageClientConfig {
    /**
     * From the AWS Docs: https://sdk.amazonaws.com/java/api/latest/software/amazon/awssdk/core/client/config/ClientOverrideConfiguration.html#apiCallTimeout()
     * The amount of time to allow the client to complete the execution of an API call.
     *
     * The value for this property configures the amount of time for the entire execution, including all retry attempts.
     * */
    private static final int S3_API_CALL_TIMEOUT_SECONDS = 15;

    /**
     * From the AWS Docs: https://sdk.amazonaws.com/java/api/latest/software/amazon/awssdk/core/client/config/ClientOverrideConfiguration.html#apiCallAttemptTimeout()
     * The amount of time to wait for the http request to complete before giving up and timing out.
     *
     * */
    private static final int S3_API_CALL_ATTEMPT_TIMEOUT_SECONDS = 5;

    private final AwsCredentialsProvider awsCredentialsProvider;
    private final DocumentStoreConfig documentStoreConfig;
    private final EncryptionClientConfig encryptionClientConfig;

    @Profile("!aws")
    @Bean
    public S3EncryptionClient localS3EncryptionClient() {
        log.warn("S3: Using local encryption without AWS KMS. Not appropriate for deployed environments!");
        // set a temporary aws region system property to satisfy the default credentials provider
        // that is triggered during initialization (before we override it with our own client definition in
        // .wrappedClient())
        System.setProperty("aws.region", documentStoreConfig.getRegion());
        S3EncryptionClient client = S3EncryptionClient.builder()
                .wrappedClient(synchronousStaticCredentialClient())
                .wrappedAsyncClient(asyncStaticCredentialClient())
                .aesKey(encryptionClientConfig.getLocalAesWrappingKey())
                .build();
        // remove temporary aws region system property
        System.clearProperty("aws.region");
        return client;
    }

    @Profile("!aws")
    @Bean
    public S3AsyncEncryptionClient localS3AsyncEncryptionClient() {
        log.warn("S3: Using local encryption without AWS KMS. Not appropriate for deployed environments!");
        // set a temporary aws region system property to satisfy the default credentials provider
        // that is triggered during initialization (before we override it with our own client definition in
        // .wrappedClient())
        System.setProperty("aws.region", documentStoreConfig.getRegion());
        S3AsyncEncryptionClient client = S3AsyncEncryptionClient.builder()
                .wrappedClient(asyncStaticCredentialClient())
                .aesKey(encryptionClientConfig.getLocalAesWrappingKey())
                .build();
        // remove temporary aws region system property
        System.clearProperty("aws.region");
        return client;
    }

    @Profile("!aws")
    @Bean
    public S3AsyncClient asyncStaticCredentialClient() {
        return S3AsyncClient.builder()
                .region(Region.of(documentStoreConfig.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(documentStoreConfig.getEndpoint()))
                .overrideConfiguration(b -> b.apiCallTimeout(Duration.ofSeconds(S3_API_CALL_TIMEOUT_SECONDS))
                        .apiCallAttemptTimeout(Duration.ofSeconds(S3_API_CALL_ATTEMPT_TIMEOUT_SECONDS)))
                .build();
    }

    @Profile("!aws")
    @Bean
    public S3Client synchronousStaticCredentialClient() {
        return S3Client.builder()
                .region(Region.of(documentStoreConfig.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(documentStoreConfig.getEndpoint()))
                .overrideConfiguration(b -> b.apiCallTimeout(Duration.ofSeconds(S3_API_CALL_TIMEOUT_SECONDS))
                        .apiCallAttemptTimeout(Duration.ofSeconds(S3_API_CALL_ATTEMPT_TIMEOUT_SECONDS)))
                .build();
    }

    @Profile("aws")
    @Bean
    public S3AsyncEncryptionClient s3AsyncEncryptionClient() {
        log.info("S3: Setting up KMS for encryption...");
        return S3AsyncEncryptionClient.builder()
                .wrappedClient(s3AssumeRoleClient())
                .cryptoMaterialsManager(encryptionClientConfig.kmsCrypto(documentStoreConfig.getKmsWrappingKeyArn()))
                .build();
    }

    @Profile("aws")
    @Bean
    public S3EncryptionClient s3SynchronousEncryptionClient() {
        log.info("S3: Setting up KMS for encryption...");
        return S3EncryptionClient.builder()
                .wrappedClient(s3SynchronousAssumeRoleClient())
                .wrappedAsyncClient(s3AssumeRoleClient())
                .cryptoMaterialsManager(encryptionClientConfig.kmsCrypto(documentStoreConfig.getKmsWrappingKeyArn()))
                .build();
    }

    @Profile("aws")
    @Bean
    public S3AsyncClient s3AssumeRoleClient() {
        return S3AsyncClient.builder()
                .region(Region.of(documentStoreConfig.getRegion()))
                .credentialsProvider(StsAssumeRoleCredentialsProvider.builder()
                        .stsClient(StsClient.builder()
                                .region(Region.of(documentStoreConfig.getRegion()))
                                .credentialsProvider(awsCredentialsProvider)
                                .build())
                        .refreshRequest(AssumeRoleRequest.builder()
                                .roleArn(documentStoreConfig.getAssumeRoleArn())
                                .roleSessionName(documentStoreConfig.getAssumeRoleSessionName())
                                .durationSeconds(Integer.parseInt(documentStoreConfig.getAssumeRoleDurationSeconds()))
                                .build())
                        .build())
                .overrideConfiguration(b -> b.apiCallTimeout(Duration.ofSeconds(S3_API_CALL_TIMEOUT_SECONDS))
                        .apiCallAttemptTimeout(Duration.ofSeconds(S3_API_CALL_ATTEMPT_TIMEOUT_SECONDS)))
                .endpointOverride(URI.create(documentStoreConfig.getEndpoint()))
                .build();
    }

    @Profile("aws")
    @Bean
    public S3Client s3SynchronousAssumeRoleClient() {
        return S3Client.builder()
                .region(Region.of(documentStoreConfig.getRegion()))
                .credentialsProvider(StsAssumeRoleCredentialsProvider.builder()
                        .stsClient(StsClient.builder()
                                .region(Region.of(documentStoreConfig.getRegion()))
                                .credentialsProvider(awsCredentialsProvider)
                                .build())
                        .refreshRequest(AssumeRoleRequest.builder()
                                .roleArn(documentStoreConfig.getAssumeRoleArn())
                                .roleSessionName(documentStoreConfig.getAssumeRoleSessionName())
                                .durationSeconds(Integer.parseInt(documentStoreConfig.getAssumeRoleDurationSeconds()))
                                .build())
                        .build())
                .overrideConfiguration(b -> b.apiCallTimeout(Duration.ofSeconds(S3_API_CALL_TIMEOUT_SECONDS))
                        .apiCallAttemptTimeout(Duration.ofSeconds(S3_API_CALL_ATTEMPT_TIMEOUT_SECONDS)))
                .endpointOverride(URI.create(documentStoreConfig.getEndpoint()))
                .build();
    }
}
