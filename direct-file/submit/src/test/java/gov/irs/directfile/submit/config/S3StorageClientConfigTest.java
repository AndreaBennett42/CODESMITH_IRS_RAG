package gov.irs.directfile.submit.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

import static org.assertj.core.api.Assertions.assertThat;

public class S3StorageClientConfigTest {
    private final ApplicationContextRunner localContextRunner = new ApplicationContextRunner()
            .withPropertyValues(
                    "aws.accessKey=test",
                    "aws.secretKey=test",
                    "aws.region=us-east-1",
                    "aws.default-credentials-provider-chain-enabled=false",
                    "submit.documentstore.endpoint=http://test.com",
                    "submit.documentstore.region=us-east-1",
                    "direct-file.local-encryption.local-wrapping-key=test")
            .withUserConfiguration(
                    S3StorageClientConfig.class, EncryptionClientConfig.class, AWSCredentialsConfig.class);

    @Test
    void localS3EncryptionClientInstantiatesWithValidProperties() {
        this.localContextRunner.run((context) ->
                assertThat(context.getBean(S3AsyncEncryptionClient.class)).isNotNull());
    }

    @Test
    void localS3EncryptionClientDoesNotInstantiateWithoutApplicableProperties() {
        localContextRunner.withPropertyValues("submit.documentstore.endpoint=").run((context) -> assertThat(context)
                .hasFailed());

        localContextRunner.withPropertyValues("submit.documentstore.region=").run((context) -> assertThat(context)
                .hasFailed());

        localContextRunner.withPropertyValues("aws.accessKey=").run((context) -> assertThat(context)
                .hasFailed());

        localContextRunner.withPropertyValues("aws.secretKey=").run((context) -> assertThat(context)
                .hasFailed());

        localContextRunner
                .withPropertyValues("direct-file.local-encryption.local-wrapping-key=")
                .run((context) -> assertThat(context).hasFailed());
    }

    @Test
    void kmsS3EncryptionClientDoesNotInstantiateWithoutApplicableProperties() {
        // testing that the aws configuration fails if you only have a local wrapping key set,
        // instead of the kms wrapping key
        ApplicationContextRunner awsContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "spring.profiles.active=aws",
                        "aws.default-credentials-provider-chain-enabled=false",
                        "aws.accessKey=test",
                        "aws.secretKey=test",
                        "aws.region=us-east-1",
                        "aws.kmsEndpoint=http://test.com",
                        "submit.documentstore.region=us-east-1",
                        "submit.documentstore.endpoint=http://test.com",
                        "submit.documentstore.assume-role-duration-seconds=60",
                        "direct-file.local-encryption.local-wrapping-key=test")
                .withUserConfiguration(
                        S3StorageClientConfig.class, EncryptionClientConfig.class, AWSCredentialsConfig.class);

        // no kms wrapping key
        awsContextRunner.run((context) -> {
            assertThat(context).hasFailed();
            assertThat(context.getStartupFailure().getMessage())
                    .contains("kmsWrappingKeyArn is marked non-null but is null");
        });
    }
}
