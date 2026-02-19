package gov.irs.directfile.api.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.services.kms.KmsClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class KmsClientConfigurationTest {

    @Test
    void kmsClientBeanCreatedWhenApplicablePropertiesAreSetWithStaticCredentialProvider() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "spring.profiles.active=aws",
                        "aws.kmsEndpoint=http://directfile.test",
                        "direct-file.aws.region=us-east-1",
                        "direct-file.aws.credentials.accessKey=test",
                        "direct-file.aws.credentials.secretKey=test",
                        "direct-file.aws.s3.endpoint=http://directfile.test",
                        "direct-file.aws.s3.assumeRoleArn=test",
                        "direct-file.aws.s3.assumeRoleDurationSeconds=0",
                        "direct-file.aws.s3.assumeRoleSessionName=test",
                        "direct-file.aws.s3.bucket=test",
                        "direct-file.aws.s3.operations-jobs-bucket=test",
                        "direct-file.aws.default-credentials-provider-chain-enabled=false")
                .withUserConfiguration(AwsCredentialsConfiguration.class, KmsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
            assertThat(context.getBean(KmsClient.class)).isNotNull();
        });
    }

    @Test
    void kmsClientBeanCreatedWhenApplicablePropertiesAreSetWithDefaultCredentialProvider() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "spring.profiles.active=aws",
                        "aws.kmsEndpoint=http://directfile.test",
                        "direct-file.aws.region=us-east-1",
                        "direct-file.aws.credentials.accessKey=test",
                        "direct-file.aws.credentials.secretKey=test",
                        "direct-file.aws.s3.endpoint=http://directfile.test",
                        "direct-file.aws.s3.assumeRoleArn=test",
                        "direct-file.aws.s3.assumeRoleDurationSeconds=0",
                        "direct-file.aws.s3.assumeRoleSessionName=test",
                        "direct-file.aws.s3.bucket=test",
                        "direct-file.aws.s3.operations-jobs-bucket=test",
                        "direct-file.aws.default-credentials-provider-chain-enabled=true")
                .withUserConfiguration(AwsCredentialsConfiguration.class, KmsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(DefaultCredentialsProvider.class);
            assertThat(context.getBean(KmsClient.class)).isNotNull();
        });
    }
}
