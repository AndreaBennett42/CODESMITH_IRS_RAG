package gov.irs.directfile.models.message;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlRequest;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlResponse;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;
import software.amazon.awssdk.services.sqs.model.SendMessageResponse;

@Slf4j
public class SqsPublisher implements Publisher {
    protected final SqsClient sqsClient;
    protected final String queueName;

    protected String queueUrl = null;

    public SqsPublisher(SqsClient sqsClient, String queueName) {
        this.sqsClient = sqsClient;
        this.queueName = queueName;
    }

    protected String getQueueUrl(String queueName) {
        GetQueueUrlRequest getQueueUrlRequest =
                GetQueueUrlRequest.builder().queueName(queueName).build();
        GetQueueUrlResponse getQueueUrlResponse;
        try {
            getQueueUrlResponse = sqsClient.getQueueUrl(getQueueUrlRequest);
        } catch (Exception e) {
            String errorMessage = "Exception calling SQS getQueueUrl: " + e.getMessage();
            log.error(errorMessage, e);
            throw new PublisherException(errorMessage, e);
        }

        if (getQueueUrlResponse.sdkHttpResponse().isSuccessful()) {
            log.info("Retrieved SQS queue url");
        } else {
            String errorMessage = "SQS getQueueUrl was unsuccessful. HTTP Status code: "
                    + getQueueUrlResponse.sdkHttpResponse().statusCode();
            log.error(errorMessage);
            throw new PublisherException(errorMessage);
        }

        return getQueueUrlResponse.queueUrl();
    }

    @Override
    public void publish(String message) {
        // Lazy load and then cache the queue URL -- having some issues trying to look it up in the constructor.
        if (StringUtils.isBlank(queueUrl)) {
            queueUrl = getQueueUrl(queueName);
        }

        SendMessageRequest sendMsgRequest = SendMessageRequest.builder()
                .queueUrl(queueUrl)
                .messageBody(message)
                .build();
        SendMessageResponse sendMessageResponse;
        try {
            sendMessageResponse = sqsClient.sendMessage(sendMsgRequest);
        } catch (Exception e) {
            String errorMessage = "Exception calling SQS sendMessage: " + e.getMessage();
            log.error(errorMessage, e);
            throw new PublisherException(errorMessage, e);
        }

        if (sendMessageResponse.sdkHttpResponse().isSuccessful()) {
            log.info("Sent message to SQS queue: {}", queueName);
        } else {
            String errorMessage = "SQS sendMessage request was unsuccessful. HTTP Status code: "
                    + sendMessageResponse.sdkHttpResponse().statusCode();
            log.error(errorMessage);
            throw new PublisherException(errorMessage);
        }
    }
}
