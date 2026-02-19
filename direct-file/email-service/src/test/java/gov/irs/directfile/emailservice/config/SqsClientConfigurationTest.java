package gov.irs.directfile.emailservice.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.services.sqs.SqsClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class SqsClientConfigurationTest {
    @Test
    void sqsClientCreatedWhenDefaultCredentialProviderChainNotEnabled() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "email-service.environment=test",
                        "email-service.messageQueue.accessKey=test",
                        "email-service.messageQueue.secretKey=test",
                        "email-service.messageQueue.default-credentials-provider-chain-enabled=false",
                        "email-service.messageQueue.send-email-queue=send-mail",
                        "email-service.messageQueue.dlq-send-email-queue=dlq-send-mail",
                        "email-service.messageQueue.endpoint=http://directfile.test",
                        "email-service.messageQueue.region=us-west-2",
                        "email-service.sender.from=test@directfile.test")
                .withUserConfiguration(AwsCredentialsConfig.class, SqsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(SqsClient.class)).isNotNull();
        });
    }

    @Test
    void sqsClientCreatedWhenDefaultCredentialProviderChainNotSet() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "email-service.environment=test",
                        "email-service.messageQueue.accessKey=test",
                        "email-service.messageQueue.secretKey=test",
                        "email-service.messageQueue.send-email-queue=send-mail",
                        "email-service.messageQueue.dlq-send-email-queue=dlq-send-mail",
                        "email-service.messageQueue.endpoint=http://directfile.test",
                        "email-service.messageQueue.region=us-west-2",
                        "email-service.sender.from=test@directfile.test")
                .withUserConfiguration(AwsCredentialsConfig.class, SqsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(SqsClient.class)).isNotNull();
        });
    }

    @Test
    void sqsClientCreatedWhenDefaultCredentialProviderChainEnabled() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "email-service.environment=test",
                        "email-service.messageQueue.accessKey=test",
                        "email-service.messageQueue.secretKey=test",
                        "email-service.messageQueue.default-credentials-provider-chain-enabled=true",
                        "email-service.messageQueue.send-email-queue=send-mail",
                        "email-service.messageQueue.dlq-send-email-queue=dlq-send-mail",
                        "email-service.messageQueue.endpoint=http://directfile.test",
                        "email-service.messageQueue.region=us-west-2",
                        "email-service.sender.from=test@directfile.test")
                .withUserConfiguration(AwsCredentialsConfig.class, SqsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(SqsClient.class)).isNotNull();
        });
    }
}
