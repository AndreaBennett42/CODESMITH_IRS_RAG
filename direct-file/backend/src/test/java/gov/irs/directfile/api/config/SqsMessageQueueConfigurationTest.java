package gov.irs.directfile.api.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.services.sqs.SqsClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class SqsMessageQueueConfigurationTest {

    @Test
    void sqsClientCreatedWhenApplicablePropertiesSet() {
        ApplicationContextRunner awsContextRunner = new ApplicationContextRunner()
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
                        "direct-file.aws.messagequeue.endpoint=http://directfile.test",
                        "direct-file.aws.messagequeue.dispatchQueue=test-queue-01",
                        "direct-file.aws.messagequeue.dlqStatusChangeQueue=test-queue-02",
                        "direct-file.aws.messagequeue.dlqSubmissionConfirmationQueue=test-queue-03",
                        "direct-file.aws.messagequeue.dlqS3NotificationEventQueue=test-queue-04",
                        "direct-file.aws.messagequeue.sendEmailQueue=test-queue-05",
                        "direct-file.aws.messagequeue.statusChangeQueue=test-queue-06",
                        "direct-file.aws.messagequeue.submissionConfirmationQueue=test-queue-07",
                        "direct-file.aws.messagequeue.s3NotificationEventQueue=test-queue-08",
                        "direct-file.aws.messagequeue.dataImportRequestQueue=test-queue-09",
                        "direct-file.aws.messagequeue.dataImportResultQueue=test-queue-10",
                        "direct-file.aws.messagequeue.dlqDataImportRequestQueue=test-queue-11",
                        "direct-file.aws.messagequeue.dlqDataImportResultQueue=test-queue-12")
                .withUserConfiguration(SqsMessageQueueConfiguration.class, AwsCredentialsConfiguration.class);

        awsContextRunner.run((context) -> {
            assertThat(context.getBean(SqsClient.class)).isNotNull();
        });
    }
}
