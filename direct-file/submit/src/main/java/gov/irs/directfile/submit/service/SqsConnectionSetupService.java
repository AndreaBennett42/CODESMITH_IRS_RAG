package gov.irs.directfile.submit.service;

import java.util.List;

import com.amazon.sqs.javamessaging.ProviderConfiguration;
import com.amazon.sqs.javamessaging.SQSConnection;
import com.amazon.sqs.javamessaging.SQSConnectionFactory;
import com.amazon.sqs.javamessaging.SQSSession;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import jakarta.jms.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.pending.PendingSubmissionMessageVersion;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.models.message.pending.payload.PendingSubmissionPayloadV1;
import gov.irs.directfile.submit.config.MessageQueueConfig;

@Slf4j
@Service
@SuppressWarnings({"PMD.AvoidReassigningParameters", "PMD.CloseResource", "PMD.UnusedPrivateMethod"})
@EnableConfigurationProperties(MessageQueueConfig.class)
public class SqsConnectionSetupService {
    private final SqsClient sqsClient;
    private final String dispatchQueue;
    private final String dispatchQueueDlq;
    private final String pendingSubmissionQueue;
    private final boolean isPendingSubmissionPublishEnabled;
    private String pendingSubmissionQueueUrl = null;
    private final ObjectMapper mapper = new ObjectMapper();
    private SQSConnection connection;

    SqsConnectionSetupService(SqsClient sqsClient, MessageQueueConfig messageQueueConfig) {
        this.sqsClient = sqsClient;
        dispatchQueue = messageQueueConfig.getDispatchQueue();
        dispatchQueueDlq = messageQueueConfig.getDlqDispatchQueue();
        pendingSubmissionQueue = messageQueueConfig.getPendingSubmissionQueue();
        isPendingSubmissionPublishEnabled = messageQueueConfig.isPendingSubmissionPublishEnabled();
    }

    public void setup(MessageListener messageListener) throws JMSException {
        SQSConnectionFactory connectionFactory = new SQSConnectionFactory(new ProviderConfiguration(), sqsClient);
        connection = connectionFactory.createConnection();

        Session session = connection.createSession(false, SQSSession.UNORDERED_ACKNOWLEDGE);
        Queue queue = session.createQueue(dispatchQueue);
        MessageConsumer consumer = session.createConsumer(queue);
        consumer.setMessageListener(messageListener);
        connection.start();
        log.info("CONNECTED TO SQS");
    }

    public void sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(
            List<TaxReturnIdAndSubmissionId> taxReturnIdAndSubmissionIds) throws JsonProcessingException {
        if (!isPendingSubmissionPublishEnabled) {
            return;
        }

        try {
            AbstractPendingSubmissionPayload payload = new PendingSubmissionPayloadV1(taxReturnIdAndSubmissionIds);
            VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> queueMessage =
                    new VersionedPendingSubmissionMessage<>(
                            payload,
                            new QueueMessageHeaders()
                                    .addHeader(
                                            MessageHeaderAttribute.VERSION,
                                            PendingSubmissionMessageVersion.V1.getVersion()));

            String jsonString = mapper.writeValueAsString(queueMessage);
            enqueueListOfSubmissionAndTaxReturnIds(jsonString);
        } catch (SqsException | JsonProcessingException e) {
            log.error("Error sending message to {}: {}", pendingSubmissionQueue, e.getMessage());
        }
    }

    private void enqueueListOfSubmissionAndTaxReturnIds(String pendingSubmissionQueueMessage) {
        if (StringUtils.isBlank(pendingSubmissionQueueUrl)) {
            pendingSubmissionQueueUrl = getQueueUrl(pendingSubmissionQueue);
        }

        SendMessageRequest sendMsgRequest = SendMessageRequest.builder()
                .queueUrl(pendingSubmissionQueueUrl)
                .messageBody(pendingSubmissionQueueMessage)
                .build();

        SendMessageResponse sendMessageResponse = sqsClient.sendMessage(sendMsgRequest);
        if (sendMessageResponse.sdkHttpResponse().isSuccessful()) {
            log.info("Sent list of tax return ids and submission ids to SQS {}", pendingSubmissionQueue);
        } else {
            log.error(
                    "SQS sendMessage request was unsuccessful. HTTP Status code: {}",
                    sendMessageResponse.sdkHttpResponse().statusCode());
        }
    }

    private String getQueueUrl(String queueName) {
        GetQueueUrlResponse getQueueUrlResponse = sqsClient.getQueueUrl(
                GetQueueUrlRequest.builder().queueName(queueName).build());

        return getQueueUrlResponse.queueUrl();
    }

    @PreDestroy
    private void cleanup() throws JMSException {
        if (connection != null) {
            connection.stop();
            log.info("SQS connection stopped");
            connection.close();
            log.info("SQS connection closed");
        }
    }
}
