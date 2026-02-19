package gov.irs.directfile.status.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.services.sqs.SqsClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class SqsClientConfigurationTest {

    @Test
    void sqsClientBeanCreatedWhenStaticCredentialProviderEnabled() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "aws.accessKey=test",
                        "aws.secretKey=test",
                        "status.message-queue.endpoint=http://directfile.test",
                        "status.message-queue.status-change-queue=test-queue-01",
                        "status.message-queue.pending-submission-queue=test-queue-02",
                        "status.message-queue.dlq-pending-submission-queue=test-queue-03",
                        "status.message-queue.region=us-west-2",
                        "status.message-queue.accessKey=test",
                        "status.message-queue.secretKey=test",
                        "aws.default-credentials-provider-chain-enabled=false")
                .withUserConfiguration(AWSCredentialsConfiguration.class, SqsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(StaticCredentialsProvider.class);
            assertThat(context.getBean(SqsClient.class)).isNotNull();
        });
    }

    @Test
    void sqsClientBeanCreatedWhenDefaultCredentialProviderEnabled() {
        ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
                .withPropertyValues(
                        "status.message-queue.endpoint=http://directfile.test",
                        "status.message-queue.status-change-queue=test-queue-01",
                        "status.message-queue.pending-submission-queue=test-queue-02",
                        "status.message-queue.dlq-pending-submission-queue=test-queue-03",
                        "status.message-queue.region=us-west-2",
                        "status.message-queue.accessKey=test",
                        "status.message-queue.secretKey=test",
                        "aws.default-credentials-provider-chain-enabled=true")
                .withUserConfiguration(AWSCredentialsConfiguration.class, SqsClientConfiguration.class);
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(AwsCredentialsProvider.class)).isInstanceOf(DefaultCredentialsProvider.class);
            assertThat(context.getBean(SqsClient.class)).isNotNull();
        });
    }
}
