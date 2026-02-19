package gov.irs.directfile.emailservice.listeners;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.jms.Message;
import jakarta.jms.MessageListener;
import jakarta.jms.TextMessage;
import lombok.extern.slf4j.Slf4j;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlRequest;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;

import gov.irs.directfile.emailservice.config.EmailServiceConfigurationProperties;
import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.listeners.handlers.email.EmailProcessingResults;
import gov.irs.directfile.emailservice.listeners.handlers.email.SendEmailV1Handler;
import gov.irs.directfile.emailservice.listeners.handlers.email.UnsupportedMessageVersionHandler;
import gov.irs.directfile.emailservice.services.EmailRecordKeepingService;
import gov.irs.directfile.emailservice.services.ISendService;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.email.SendEmailMessageVersion;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

@Slf4j
@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP2", "CT_CONSTRUCTOR_THROW", "REC_CATCH_EXCEPTION"},
        justification =
                "SQS Client is not exposed externally; Java 21 update; No requirements for individual exception handling at this time")
@SuppressWarnings("PMD.UselessParentheses")
public class SendEmailMessageQueueListener implements MessageListener {
    private final SqsClient sqsClient;
    private final EmailRecordKeepingService emailRecordKeepingService;
    private final String sendEmailQueueUrl;
    // TODO Consider using the auto-configured ObjectMapper -
    // https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/#features.json.jackson
    private final ObjectMapper mapper = new ObjectMapper();
    private final int resendDelaySeconds;

    private final SendEmailMessageRouter sendEmailMessageRouter;

    public SendEmailMessageQueueListener(
            EmailServiceConfigurationProperties configProps,
            ISendService sender,
            SqsClient sqsClient,
            EmailRecordKeepingService emailRecordKeepingService) {

        this.sqsClient = sqsClient;
        this.emailRecordKeepingService = emailRecordKeepingService;
        this.sendEmailQueueUrl = sqsClient
                .getQueueUrl(GetQueueUrlRequest.builder()
                        .queueName(configProps.getMessageQueue().getSendEmailQueue())
                        .build())
                .queueUrl();
        this.resendDelaySeconds = configProps.getResendDelaySeconds();

        this.sendEmailMessageRouter = new SendEmailMessageRouter(
                new UnsupportedMessageVersionHandler(), new SendEmailV1Handler(sender, emailRecordKeepingService));
    }

    @Override
    public void onMessage(Message message) {
        String rawText = "";
        log.info("Received a send mail request");
        try {
            rawText = ((TextMessage) message).getText();

            VersionedSendEmailMessage<AbstractSendEmailPayload> versionedSubmissionConfirmationMessage =
                    mapper.readValue(rawText, new TypeReference<>() {});
            EmailProcessingResults results =
                    sendEmailMessageRouter.handleSendEmailMessage(versionedSubmissionConfirmationMessage);
            if (results.countToSend() != results.countSent()) {
                // if an email is not sent, we shouldn't block acknowledgement of the entire message otherwise we
                // would send multiple messages to many users just because a single message in the batch wasn't sent

                if (results.countSent() == 0) {
                    log.error(String.format(
                            "Zero emails sent in batch size of %s, likely due to dropped connection to email relay. Returning and not acknowledging to force re-enqueue",
                            results.countToSend()));
                } else {
                    log.error("{} of {} emails were sent.", results.countSent(), results.countToSend());
                }

                if (results.countMessagingException() > 0) {
                    log.error(
                            "{} of {} emails failed on creation.",
                            results.countMessagingException(),
                            results.countToSend());
                }

                // add emails to resend back to SQS
                try {
                    sqsClient.sendMessage(convertSendEmailsToSendMessageRequest(results.emailsToResend()));
                } catch (JsonProcessingException e) {
                    // failed to write JSON message for SendMessageRequest
                    log.error(
                            "Failed to resend {} emails.",
                            results.emailsToResend().size());
                }

            } else {
                log.info("All emails sent");
            }

            message.acknowledge();

        } catch (Exception e) {
            if (rawText != null) {
                log.error("Unable to parse the value of message");
            } else {
                log.error("Unable to process a message from the queue, likely not a text message", e);
            }
        }
    }

    protected SendMessageRequest convertSendEmailsToSendMessageRequest(List<SendEmail> sendEmails)
            throws JsonProcessingException {
        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> emailSqsMessages = new EnumMap<>(HtmlTemplate.class);
        for (SendEmail sendEmail : sendEmails) {
            if (!emailSqsMessages.containsKey(sendEmail.getEmailType())) {
                emailSqsMessages.put(sendEmail.getEmailType(), new ArrayList<>());
            }
            emailSqsMessages
                    .get(sendEmail.getEmailType())
                    .add(new SendEmailQueueMessageBody(
                            sendEmail.getRecipientEmailAddress(),
                            sendEmail.getLanguageCode(),
                            sendEmail.getTaxReturnId(),
                            sendEmail.getSubmissionId(),
                            sendEmail.getUserId(),
                            sendEmail.getId(),
                            sendEmail.getContext()));
        }

        AbstractSendEmailPayload payload = new SendEmailPayloadV1(emailSqsMessages);
        VersionedSendEmailMessage<AbstractSendEmailPayload> queueMessage = new VersionedSendEmailMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, SendEmailMessageVersion.V1.getVersion()));

        try {
            return SendMessageRequest.builder()
                    .queueUrl(sendEmailQueueUrl)
                    .messageBody(mapper.writeValueAsString(queueMessage))
                    .delaySeconds(resendDelaySeconds)
                    .build();
        } catch (JsonProcessingException e) {
            log.error(
                    "Failed to serialize Map<HtmlTemplate, List<SendEmailQueueMessageBody>> as JSON; unable to build SendMessageRequest");
            throw e;
        }
    }
}
