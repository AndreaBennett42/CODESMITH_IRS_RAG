package gov.irs.directfile.api.taxreturn.submissions;

import jakarta.jms.JMSException;
import jakarta.jms.Message;
import jakarta.jms.MessageListener;
import jakarta.jms.TextMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;

@Slf4j
@Service
@ConditionalOnProperty(value = "direct-file.aws.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class S3NotificationEventQueueListenerService implements MessageListener {

    private String queueName;
    private final S3NotificationEventService s3NotificationEventService;

    S3NotificationEventQueueListenerService(
            MessageQueueConfigurationProperties messageQueueConfigurationProperties,
            S3NotificationEventService s3NotificationEventService) {
        this.queueName = messageQueueConfigurationProperties.getS3NotificationEventQueue();
        this.s3NotificationEventService = s3NotificationEventService;
    }

    @Override
    public void onMessage(Message message) {
        log.info("onMessage called ({})", queueName);

        try {
            String rawText = ((TextMessage) message).getText();
            s3NotificationEventService.handleS3NotificationEvent(rawText);

        } catch (Exception e) {
            log.error(
                    "Error parsing SQS message in {} {}",
                    queueName,
                    e.getClass().getName());
        }

        try {
            message.acknowledge();
        } catch (JMSException e) {
            throw new RuntimeException(e);
        }
    }
}
