package gov.irs.directfile.status.services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.jms.Message;
import jakarta.jms.MessageListener;
import jakarta.jms.TextMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.status.config.MessageQueueConfiguration;

@Slf4j
@Service
@ConditionalOnProperty(value = "status.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfiguration.class)
public class MessageQueueListenerService implements MessageListener {
    private final String pendingSubmissionQueue;
    private final PendingSubmissionMessageRouter pendingSubmissionMessageRouter;
    private final SubmissionConfirmationMessageRouter submissionConfirmationMessageRouter;
    private final ObjectMapper objectMapper = new ObjectMapper();

    MessageQueueListenerService(
            MessageQueueConfiguration messageQueueConfiguration,
            PendingSubmissionMessageRouter pendingSubmissionMessageRouter,
            SubmissionConfirmationMessageRouter submissionConfirmationMessageRouter) {
        this.pendingSubmissionQueue = messageQueueConfiguration.getPendingSubmissionQueue();
        this.pendingSubmissionMessageRouter = pendingSubmissionMessageRouter;
        this.submissionConfirmationMessageRouter = submissionConfirmationMessageRouter;
    }

    @Override
    public void onMessage(Message message) {
        log.info("onMessage called ({})", pendingSubmissionQueue);

        String rawText = "";
        try {
            rawText = ((TextMessage) message).getText();

            try {
                // First try to deserialize as a VersionedPendingSubmissionMessage (what SQS would send)
                log.info("Trying to deserialize message to VersionedPendingSubmissionMessage");
                VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> versionedPendingSubmissionMessage =
                        objectMapper.readValue(rawText, new TypeReference<>() {});
                pendingSubmissionMessageRouter.handlePendingSubmissionMessage(versionedPendingSubmissionMessage);
            } catch (JsonProcessingException e) {
                // Otherwise, try to deserialize as a VersionedSubmissionConfirmationMessage (what SNS would send)
                log.info("Trying to deserialize message to VersionedSubmissionConfirmationMessage");
                VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload>
                        versionedSubmissionConfirmationMessage =
                                objectMapper.readValue(rawText, new TypeReference<>() {});
                submissionConfirmationMessageRouter.handleSubmissionConfirmationMessage(
                        versionedSubmissionConfirmationMessage);
            }

            message.acknowledge();
        } catch (Exception e) {
            log.error(
                    "Error saving Pending objects in database. Re-queueing list of submissionIds: {}. Error: {}",
                    rawText,
                    e.getMessage());
        }
    }
}
