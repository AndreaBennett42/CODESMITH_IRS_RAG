package gov.irs.directfile.api.taxreturn.submissions;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.jms.Message;
import jakarta.jms.MessageListener;
import jakarta.jms.TextMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;

@Service
@Slf4j
@ConditionalOnProperty(value = "direct-file.aws.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@SuppressWarnings("PMD.ExceptionAsFlowControl")
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class StatusQueueListenerService implements MessageListener {
    private final String queueName;
    private final ObjectMapper objectMapper;

    private final StatusChangeMessageRouter statusChangeMessageRouter;

    public StatusQueueListenerService(
            MessageQueueConfigurationProperties messageQueueConfigurationProperties,
            StatusChangeMessageRouter statusChangeMessageRouter,
            ObjectMapper objectMapper) {
        this.queueName = messageQueueConfigurationProperties.getStatusChangeQueue();
        this.statusChangeMessageRouter = statusChangeMessageRouter;
        this.objectMapper = objectMapper;
    }

    @Override
    public void onMessage(Message message) {
        log.info("onMessage called ({})", queueName);

        try {
            String rawText = ((TextMessage) message).getText();
            // don't log rawText
            // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/5306

            VersionedStatusChangeMessage<AbstractStatusChangePayload> versionedStatusChangeMessage =
                    objectMapper.readValue(rawText, new TypeReference<>() {});
            statusChangeMessageRouter.handleStatusChangeMessage(versionedStatusChangeMessage);

            message.acknowledge();
        } catch (Exception e) {
            log.error(
                    "Error parsing SQS message in {}, {}, {}",
                    queueName,
                    e.getClass().getName(),
                    e.getMessage(),
                    e);
        }
    }
}
