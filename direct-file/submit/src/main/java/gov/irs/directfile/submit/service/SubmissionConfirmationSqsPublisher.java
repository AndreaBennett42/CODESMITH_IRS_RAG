package gov.irs.directfile.submit.service;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;

import gov.irs.directfile.models.message.SqsPublisher;
import gov.irs.directfile.submit.config.MessageQueueConfig;

@Service
@ConditionalOnProperty(value = "submit.messagequeue.submission-confirmation-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfig.class)
public class SubmissionConfirmationSqsPublisher extends SqsPublisher implements SubmissionConfirmationPublisher {
    public SubmissionConfirmationSqsPublisher(SqsClient sqsClient, MessageQueueConfig messageQueueConfig) {
        super(sqsClient, messageQueueConfig.getSubmissionConfirmationQueue());
    }
}
