package gov.irs.directfile.stateapi.configuration;

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
                        "aws.s3.accessKey=test",
                        "aws.s3.secretKey=test",
                        "aws.s3.default-credentials-provider-chain-enabled=false")
                .withUserConfiguration(AwsCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
        });
    }

    @Test
    void staticCredentialProviderCreatedWhenApplicablePropertyIsNotSet() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues("aws.s3.accessKey=test", "aws.s3.secretKey=test")
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
                        "aws.s3.accessKey=test",
                        "aws.s3.secretKey=test",
                        "aws.s3.default-credentials-provider-chain-enabled=true")
                .withUserConfiguration(AwsCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(DefaultCredentialsProvider.class);
        });
    }
}
