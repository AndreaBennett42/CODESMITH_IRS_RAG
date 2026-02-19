package gov.irs.directfile.api.dispatch;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;
import gov.irs.directfile.models.Dispatch;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.dispatch.DispatchMessageVersion;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.dispatch.payload.DispatchPayloadV1;

@Service
@Slf4j
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class DispatchQueueService {
    private final String queueName;
    private String queueUrl;
    private final SqsClient sqsClient;
    private final ObjectMapper mapper = new ObjectMapper();

    public DispatchQueueService(
            SqsClient sqsClient, MessageQueueConfigurationProperties messageQueueConfigurationProperties) {
        this.sqsClient = sqsClient;
        this.queueName = messageQueueConfigurationProperties.getDispatchQueue();

        JavaTimeModule module = new JavaTimeModule();
        DateFormat df = new SimpleDateFormat("yyyy-dd-MM HH:mm:ss", Locale.US);
        mapper.setDateFormat(df);
        mapper.registerModule(module);
    }

    public void enqueue(Dispatch dispatch) {
        try {
            AbstractDispatchPayload payload = new DispatchPayloadV1(dispatch);
            VersionedDispatchMessage<AbstractDispatchPayload> queueMessage = new VersionedDispatchMessage<>(
                    payload,
                    new QueueMessageHeaders()
                            .addHeader(MessageHeaderAttribute.VERSION, DispatchMessageVersion.V1.getVersion()));

            String dispatchJsonString = mapper.writeValueAsString(queueMessage);

            if (queueUrl == null || StringUtils.isBlank(queueUrl)) {
                GetQueueUrlResponse getQueueUrlResponse = sqsClient.getQueueUrl(
                        GetQueueUrlRequest.builder().queueName(queueName).build());
                queueUrl = getQueueUrlResponse.queueUrl();
            }

            // tax return id must not be null, otherwise NPE in SendMessageRequest builder.
            MessageAttributeValue attrTaxReturnId = MessageAttributeValue.builder()
                    .dataType("String")
                    .stringValue(dispatch.getTaxReturnId().toString())
                    .build();

            SendMessageRequest sendMsgRequest = SendMessageRequest.builder()
                    .queueUrl(queueUrl)
                    .messageAttributes(Map.of("TAX-RETURN-ID", attrTaxReturnId))
                    .messageBody(dispatchJsonString)
                    .build();

            SendMessageResponse sendMessageResponse = sqsClient.sendMessage(sendMsgRequest);
            String messageId = sendMessageResponse != null ? sendMessageResponse.messageId() : "";
            int statusCode = sendMessageResponse != null && sendMessageResponse.sdkHttpResponse() != null
                    ? sendMessageResponse.sdkHttpResponse().statusCode()
                    : 0;

            log.info("SQS Message sent with messageId: {} and httpStatusCode: {}", messageId, statusCode);
        } catch (QueueDoesNotExistException | InvalidMessageContentsException e) {
            log.error("Error sending message to SQS: {}", e.getMessage());
        } catch (Exception e) {
            log.error("Unknown exception in SQS dispatch from class: {}", e.getClass());
        }
    }
}
