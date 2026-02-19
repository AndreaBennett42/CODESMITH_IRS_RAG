package gov.irs.directfile.models.message;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.http.SdkHttpResponse;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SqsPublisherTest {
    private SqsPublisher sqsPublisher;

    @Mock
    private SqsClient sqsClient;

    @Mock
    private GetQueueUrlResponse getQueueUrlResponse;

    @Mock
    private SdkHttpResponse sdkHttpResponseForGetQueueUrl;

    @Mock
    private SendMessageResponse sendMessageResponse;

    @Mock
    private SdkHttpResponse sdkHttpResponseForSendMessage;

    private final String queueName = "some-queue";
    private final String queueUrl = "http://localhost:4566/000000000000/some-queue";
    private final String message = "hello";

    @BeforeEach
    public void setup() {
        sqsPublisher = new SqsPublisher(sqsClient, queueName);
    }

    @Test
    public void publish_success() {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn(queueUrl);
        when(getQueueUrlResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForGetQueueUrl);
        when(sdkHttpResponseForGetQueueUrl.isSuccessful()).thenReturn(true);

        when(sqsClient.sendMessage(any(SendMessageRequest.class))).thenReturn(sendMessageResponse);
        when(sendMessageResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForSendMessage);
        when(sdkHttpResponseForSendMessage.isSuccessful()).thenReturn(true);

        sqsPublisher.publish(message);

        GetQueueUrlRequest getQueueUrlRequest =
                GetQueueUrlRequest.builder().queueName(queueName).build();

        ArgumentCaptor<GetQueueUrlRequest> urlArgumentCaptor = ArgumentCaptor.forClass(GetQueueUrlRequest.class);
        verify(sqsClient, times(1)).getQueueUrl(urlArgumentCaptor.capture());
        assertEquals(getQueueUrlRequest, urlArgumentCaptor.getValue());

        SendMessageRequest sendMessageRequest = SendMessageRequest.builder()
                .queueUrl(queueUrl)
                .messageBody(message)
                .build();

        ArgumentCaptor<SendMessageRequest> sendArgumentCaptor = ArgumentCaptor.forClass(SendMessageRequest.class);
        verify(sqsClient, times(1)).sendMessage(sendArgumentCaptor.capture());
        assertEquals(sendMessageRequest, sendArgumentCaptor.getValue());
    }

    @Test
    public void publish_getQueueUrlFails() {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class)))
                .thenThrow(QueueDoesNotExistException.builder().build());

        assertThrows(PublisherException.class, () -> sqsPublisher.publish(message));
        verify(sqsClient, times(1)).getQueueUrl(any(GetQueueUrlRequest.class));
        verify(sqsClient, never()).sendMessage(any(SendMessageRequest.class));
    }

    @Test
    public void publish_getQueueUrlResponseFails() {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForGetQueueUrl);
        when(sdkHttpResponseForGetQueueUrl.isSuccessful()).thenReturn(false);

        assertThrows(PublisherException.class, () -> sqsPublisher.publish(message));
        verify(sqsClient, times(1)).getQueueUrl(any(GetQueueUrlRequest.class));
        verify(sqsClient, never()).sendMessage(any(SendMessageRequest.class));
    }

    @Test
    public void publish_sendMessageFails() {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn(queueUrl);
        when(getQueueUrlResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForGetQueueUrl);
        when(sdkHttpResponseForGetQueueUrl.isSuccessful()).thenReturn(true);

        when(sqsClient.sendMessage(any(SendMessageRequest.class)))
                .thenThrow(InvalidMessageContentsException.builder().build());

        assertThrows(PublisherException.class, () -> sqsPublisher.publish(message));
        verify(sqsClient, times(1)).getQueueUrl(any(GetQueueUrlRequest.class));
        verify(sqsClient, times(1)).sendMessage(any(SendMessageRequest.class));
    }

    @Test
    public void publish_sendMessageResponseFails() {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn(queueUrl);
        when(getQueueUrlResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForGetQueueUrl);
        when(sdkHttpResponseForGetQueueUrl.isSuccessful()).thenReturn(true);

        when(sqsClient.sendMessage(any(SendMessageRequest.class))).thenReturn(sendMessageResponse);
        when(sendMessageResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForSendMessage);
        when(sdkHttpResponseForSendMessage.isSuccessful()).thenReturn(false);

        assertThrows(PublisherException.class, () -> sqsPublisher.publish(message));
        verify(sqsClient, times(1)).getQueueUrl(any(GetQueueUrlRequest.class));
        verify(sqsClient, times(1)).sendMessage(any(SendMessageRequest.class));
    }

    @Test
    public void publish_urlIsCached() {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn(queueUrl);
        when(getQueueUrlResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForGetQueueUrl);
        when(sdkHttpResponseForGetQueueUrl.isSuccessful()).thenReturn(true);

        when(sqsClient.sendMessage(any(SendMessageRequest.class))).thenReturn(sendMessageResponse);
        when(sendMessageResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForSendMessage);
        when(sdkHttpResponseForSendMessage.isSuccessful()).thenReturn(true);

        // Call publish twice and make sure URL is only requested once (cached after first publish call).
        sqsPublisher.publish(message);
        sqsPublisher.publish(message);

        verify(sqsClient, times(1)).getQueueUrl(any(GetQueueUrlRequest.class));
    }
}
