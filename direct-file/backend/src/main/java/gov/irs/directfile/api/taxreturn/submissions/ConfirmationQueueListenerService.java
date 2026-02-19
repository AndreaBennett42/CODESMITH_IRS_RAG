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
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;

@Slf4j
@Service
@ConditionalOnProperty(value = "direct-file.aws.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class ConfirmationQueueListenerService implements MessageListener {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private String queueName;
    private final SubmissionConfirmationMessageRouter submissionConfirmationMessageRouter;

    ConfirmationQueueListenerService(
            MessageQueueConfigurationProperties messageQueueConfigurationProperties,
            SubmissionConfirmationMessageRouter submissionConfirmationMessageRouter) {
        this.queueName = messageQueueConfigurationProperties.getSubmissionConfirmationQueue();
        this.submissionConfirmationMessageRouter = submissionConfirmationMessageRouter;
    }

    @Override
    public void onMessage(Message message) {
        log.info("onMessage called ({})", queueName);

        try {
            String rawText = ((TextMessage) message).getText();
            log.info("Received SQS Message: {}", rawText);

            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload>
                    versionedSubmissionConfirmationMessage = objectMapper.readValue(rawText, new TypeReference<>() {});
            submissionConfirmationMessageRouter.handleSubmissionConfirmationMessage(
                    versionedSubmissionConfirmationMessage);

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
