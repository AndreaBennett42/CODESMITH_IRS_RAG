package gov.irs.directfile.api.taxreturn.submissions;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlRequest;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlResponse;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.email.SendEmailMessageVersion;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

@Slf4j
@Service
@ConditionalOnProperty(value = "direct-file.aws.messageQueue.sqs-message-sending-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class SendEmailQueueService {
    private final SqsClient sqsClient;
    private String sendEmailQueueUrl = "";
    private final String sendEmailQueue;
    // TODO Consider using the auto-configured ObjectMapper -
    // https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/#features.json.jackson
    private final ObjectMapper mapper = new ObjectMapper();

    SendEmailQueueService(
            SqsClient sqsClient, MessageQueueConfigurationProperties messageQueueConfigurationProperties) {
        this.sqsClient = sqsClient;
        this.sendEmailQueue = messageQueueConfigurationProperties.getSendEmailQueue();
    }

    public void enqueue(Map<HtmlTemplate, List<SendEmailQueueMessageBody>> emailSqsMessages) {
        try {
            AbstractSendEmailPayload payload = new SendEmailPayloadV1(emailSqsMessages);
            VersionedSendEmailMessage<AbstractSendEmailPayload> queueMessage = new VersionedSendEmailMessage<>(
                    payload,
                    new QueueMessageHeaders()
                            .addHeader(MessageHeaderAttribute.VERSION, SendEmailMessageVersion.V1.getVersion()));

            String message = mapper.writeValueAsString(queueMessage);
            sendQueueMessage(message);
        } catch (JsonProcessingException e) {
            log.error("Malformed JSON in formatting to be sent on SQS: ", e);
        }
    }

    // TODO Consider refactoring logic so it can be shared between applications such as the email service; recommend
    // combining logic from #enqueue and returning a SendMessageRequest to pass to sqsClient#sendMessage with #enqueue
    // becoming a wrapper for the sqsClient#sendMessage call
    private void sendQueueMessage(String message) {
        try {
            // TODO move this block to the constructor so sendEmailQueueUrl can be made final
            if (sendEmailQueueUrl == null || StringUtils.isBlank(sendEmailQueueUrl)) {
                GetQueueUrlResponse getQueueUrlResponse = sqsClient.getQueueUrl(
                        GetQueueUrlRequest.builder().queueName(sendEmailQueue).build());
                sendEmailQueueUrl = getQueueUrlResponse.queueUrl();
            }

            log.info("Attempting to send send email message to SQS");
            SendMessageRequest sendMsgRequest = SendMessageRequest.builder()
                    .queueUrl(sendEmailQueueUrl)
                    .messageBody(message)
                    .build();

            sqsClient.sendMessage(sendMsgRequest);
            log.info("Sent send email message to SQS");
        } catch (Exception error) {
            log.error("Error sending send email message to SQS: ", error);
            // TODO: Implement meaningful error handing
        }
    }
}
