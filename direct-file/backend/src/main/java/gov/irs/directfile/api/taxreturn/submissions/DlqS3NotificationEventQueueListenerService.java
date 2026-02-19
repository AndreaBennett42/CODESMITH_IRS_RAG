package gov.irs.directfile.api.taxreturn.submissions;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
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
@SuppressFBWarnings(value = "CT_CONSTRUCTOR_THROW", justification = "Java 21 update")
public class DlqS3NotificationEventQueueListenerService implements MessageListener {

    private final String dlqS3NotificationEventQueueListenerServiceQueue;
    private final String s3NotificationEventQueueListenerServiceQueue;

    public DlqS3NotificationEventQueueListenerService(
            MessageQueueConfigurationProperties messageQueueConfigurationProperties) {
        this.dlqS3NotificationEventQueueListenerServiceQueue =
                messageQueueConfigurationProperties.getDlqS3NotificationEventQueue();
        this.s3NotificationEventQueueListenerServiceQueue =
                messageQueueConfigurationProperties.getS3NotificationEventQueue();
    }

    @Override
    public void onMessage(Message message) {
        log.info("onMessage called (DLQ: {})", dlqS3NotificationEventQueueListenerServiceQueue);

        try {
            String rawText = ((TextMessage) message).getText();

            log.error(
                    "DLQ: {}, Source Queue: {}, Backend App failed to process s3 notification event queue, rawText: {}",
                    dlqS3NotificationEventQueueListenerServiceQueue,
                    s3NotificationEventQueueListenerServiceQueue,
                    rawText);
        } catch (JMSException ex) {
            log.error(
                    "DLQ: {}, Source Queue: {}, Backend App failed to retrieve message in DLQ, exception: {}, error: {}",
                    dlqS3NotificationEventQueueListenerServiceQueue,
                    s3NotificationEventQueueListenerServiceQueue,
                    ex.getClass().getName(),
                    ex.getMessage());
        }
    }
}
