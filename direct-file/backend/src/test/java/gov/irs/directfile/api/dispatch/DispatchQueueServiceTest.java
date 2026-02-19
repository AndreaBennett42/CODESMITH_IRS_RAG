package gov.irs.directfile.api.dispatch;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlRequest;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlResponse;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;
import gov.irs.directfile.models.Dispatch;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.dispatch.DispatchMessageVersion;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.dispatch.payload.DispatchPayloadV1;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DispatchQueueServiceTest {
    DispatchQueueService dispatchQueueService;

    @Mock
    SqsClient sqs;

    @Mock
    MessageQueueConfigurationProperties messageQueueConfigurationProperties;

    @Mock
    GetQueueUrlResponse getQueueUrlResponse;

    String queueUrl = "http://localhost:4566/000000000000/dispatch-queue";

    @BeforeEach
    public void setup() {
        dispatchQueueService = new DispatchQueueService(sqs, messageQueueConfigurationProperties);
    }

    @Test
    void enqueue_createsJsonStringFromDispatchObjectAndSendsItAsAMessageOnSqs() throws JsonProcessingException {
        Dispatch dispatch = Dispatch.testObjectFactory();

        AbstractDispatchPayload payload = new DispatchPayloadV1(dispatch);
        VersionedDispatchMessage<AbstractDispatchPayload> queueMessage = new VersionedDispatchMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, DispatchMessageVersion.V1.getVersion()));

        ObjectMapper mapper = new ObjectMapper();
        String dispatchJsonString = mapper.writeValueAsString(queueMessage);

        when(sqs.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn(queueUrl);

        dispatchQueueService.enqueue(dispatch);

        ArgumentCaptor<SendMessageRequest> captor = ArgumentCaptor.forClass(SendMessageRequest.class);
        verify(sqs).sendMessage(captor.capture());
        SendMessageRequest sendMessageRequest = captor.getValue();

        assertEquals(dispatchJsonString, sendMessageRequest.messageBody());
    }
}
