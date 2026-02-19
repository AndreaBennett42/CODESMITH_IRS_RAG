package gov.irs.directfile.status.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;

import static org.assertj.core.api.Assertions.assertThat;

class AWSCredentialsConfigurationTest {

    @Test
    void staticCredentialProviderCreatedWhenApplicablePropertySetToFalse() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "aws.accessKey=test",
                        "aws.secretKey=test",
                        "aws.default-credentials-provider-chain-enabled=false")
                .withUserConfiguration(AWSCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
        });
    }

    @Test
    void staticCredentialProviderCreatedWhenApplicablePropertyIsNotSet() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues("aws.accessKey=test", "aws.secretKey=test")
                .withUserConfiguration(AWSCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
        });
    }

    @Test
    void defaultCredentialProviderCreatedWhenApplicablePropertySetToTrue() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "aws.accessKey=test",
                        "aws.secretKey=test",
                        "aws.default-credentials-provider-chain-enabled=true")
                .withUserConfiguration(AWSCredentialsConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isNotNull();
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(DefaultCredentialsProvider.class);
        });
    }
}
