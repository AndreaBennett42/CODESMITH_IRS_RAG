package gov.irs.directfile.submit.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class SnsClientConfigurationTest {

    @Test
    void snsClientBeanCreatedWhenStaticCredentialProviderEnabled() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "aws.accessKey=test",
                        "aws.secretKey=test",
                        "submit.sns.endpoint=http://directfile.test",
                        "submit.sns.region=us-west-2",
                        "submit.sns.submission-confirmation-publish-enabled=true",
                        "aws.default-credentials-provider-chain-enabled=false")
                .withUserConfiguration(AWSCredentialsConfig.class, SnsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
            assertThat(context.getBean(SnsClientConfiguration.class)).isNotNull();
        });
    }

    @Test
    void snsClientBeanCreatedWhenDefaultCredentialProviderEnabled() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "submit.sns.endpoint=http://directfile.test",
                        "submit.sns.region=us-west-2",
                        "submit.sns.submission-confirmation-publish-enabled=true",
                        "aws.default-credentials-provider-chain-enabled=true")
                .withUserConfiguration(AWSCredentialsConfig.class, SnsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(DefaultCredentialsProvider.class);
            assertThat(context.getBean(SnsClientConfiguration.class)).isNotNull();
        });
    }
}
