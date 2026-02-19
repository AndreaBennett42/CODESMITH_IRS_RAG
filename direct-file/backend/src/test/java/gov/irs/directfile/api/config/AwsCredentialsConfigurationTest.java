package gov.irs.directfile.api.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class AwsCredentialsConfigurationTest {

    @Test
    void staticCredentialProviderCreatedWhenApplicablePropertySetToFalse() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
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
                .withUserConfiguration(AwsCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
        });
    }

    @Test
    void staticCredentialProviderCreatedWhenApplicablePropertyIsNotSet() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "direct-file.aws.region=us-east-1",
                        "direct-file.aws.credentials.accessKey=test",
                        "direct-file.aws.credentials.secretKey=test",
                        "direct-file.aws.s3.endpoint=http://directfile.test",
                        "direct-file.aws.s3.assumeRoleArn=test",
                        "direct-file.aws.s3.assumeRoleDurationSeconds=0",
                        "direct-file.aws.s3.assumeRoleSessionName=test",
                        "direct-file.aws.s3.operations-jobs-bucket=test",
                        "direct-file.aws.s3.bucket=test")
                .withUserConfiguration(AwsCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
        });
    }

    @Test
    void defaultCredentialProviderCreatedWhenApplicablePropertySetToTrue() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
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
                .withUserConfiguration(AwsCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(DefaultCredentialsProvider.class);
        });
    }
}
