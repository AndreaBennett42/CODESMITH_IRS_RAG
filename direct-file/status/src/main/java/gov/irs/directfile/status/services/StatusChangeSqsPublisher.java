package gov.irs.directfile.status.services;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;

import gov.irs.directfile.models.message.SqsPublisher;
import gov.irs.directfile.status.config.MessageQueueConfiguration;

@Service
@ConditionalOnProperty(value = "status.messageQueue.status-change-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfiguration.class)
public class StatusChangeSqsPublisher extends SqsPublisher implements StatusChangePublisher {
    public StatusChangeSqsPublisher(SqsClient sqsClient, MessageQueueConfiguration messageQueueConfiguration) {
        super(sqsClient, messageQueueConfiguration.getStatusChangeQueue());
    }
}
