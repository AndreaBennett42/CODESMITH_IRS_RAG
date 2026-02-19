package gov.irs.directfile.api.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.encryption.s3.S3EncryptionClient;

import static org.assertj.core.api.Assertions.assertThat;

class S3StorageConfigTest {

    @Test
    void s3LocalEncryptionClientInstantiatesWithValidProperties() {
        // s3ConfigurationProperties and LocalEncryptionConfigurationProperties are @Validated so all fields
        // are required;
        final ApplicationContextRunner localContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "direct-file.aws.credentials.accessKey=test",
                        "direct-file.aws.credentials.secretKey=test",
                        "direct-file.aws.region=us-east-1",
                        "direct-file.aws.s3.endpoint=http://test.com",
                        "direct-file.aws.s3.assumeRoleArn=test",
                        "direct-file.aws.s3.assumeRoleDurationSeconds=0",
                        "direct-file.aws.s3.assumeRoleSessionName=test",
                        "direct-file.aws.s3.kmsWrappingKeyArn=test",
                        "direct-file.aws.s3.bucket=test",
                        "direct-file.aws.s3.operations-jobs-bucket=test",
                        "direct-file.aws.s3.environment-prefix=test",
                        "direct-file.local-encryption.local-wrapping-key=wjI02W2sBT1Q9P9iGTqkyEwFme4l04uz7nUYqXsntQU=")
                .withUserConfiguration(S3StorageConfig.class, AwsCredentialsConfiguration.class);

        localContextRunner.run(context -> {
            assertThat(context.getBean(S3EncryptionClient.class)).isNotNull();
        });
    }

    @Test
    void localS3EncryptionClientDoesNotInstantiateWithoutApplicableProperties() {
        final ApplicationContextRunner localContextRunner =
                new ApplicationContextRunner().withUserConfiguration(S3StorageConfig.class);

        localContextRunner.run(context -> {
            assertThat(context).hasFailed();
        });
    }

    @Test
    void s3EncryptionClientDoesNotInstantiateWithoutApplicablePropertiesAndProfileAws() {
        final ApplicationContextRunner localContextRunner = new ApplicationContextRunner()
                .withPropertyValues("spring.profiles.active=aws")
                .withUserConfiguration(S3StorageConfig.class);

        localContextRunner.run(context -> {
            assertThat(context).hasFailed();
        });
    }
}
