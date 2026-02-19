package gov.irs.directfile.submit.service;

import java.util.List;
import java.util.UUID;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.http.SdkHttpResponse;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlRequest;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlResponse;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;
import software.amazon.awssdk.services.sqs.model.SendMessageResponse;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.pending.PendingSubmissionMessageVersion;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.models.message.pending.payload.PendingSubmissionPayloadV1;
import gov.irs.directfile.submit.config.MessageQueueConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SqsConnectionSetupServiceTest {
    SqsConnectionSetupService sqsConnectionSetupService;

    @Mock
    SqsClient sqsClient;

    @Mock
    GetQueueUrlResponse getQueueUrlResponse;

    @Mock
    MessageQueueConfig messageQueueConfig;

    String pendingSubmissionQueueUrl = "http://localhost:4566/000000000000/pending-submission-queue";

    @BeforeEach
    public void setup() {
        // Turn on pending submission queue publishing by default
        when(messageQueueConfig.isPendingSubmissionPublishEnabled()).thenReturn(true);
        sqsConnectionSetupService = new SqsConnectionSetupService(sqsClient, messageQueueConfig);
    }

    @Test
    public void sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue_sendsListOfStringsUsingSqsClient()
            throws JsonProcessingException {
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn(pendingSubmissionQueueUrl);
        SendMessageResponse mockSendMessageResponse = Mockito.mock(SendMessageResponse.class);
        SdkHttpResponse mockSdkHttpResponse = Mockito.mock(SdkHttpResponse.class);
        when(sqsClient.sendMessage((SendMessageRequest) any())).thenReturn(mockSendMessageResponse);
        when(mockSendMessageResponse.sdkHttpResponse()).thenReturn(mockSdkHttpResponse);
        when(mockSdkHttpResponse.isSuccessful()).thenReturn(true);

        TaxReturnIdAndSubmissionId taxReturnIdAndSubmissionId1 =
                new TaxReturnIdAndSubmissionId(UUID.randomUUID(), "111111");
        TaxReturnIdAndSubmissionId taxReturnIdAndSubmissionId2 =
                new TaxReturnIdAndSubmissionId(UUID.randomUUID(), "222222");

        List<TaxReturnIdAndSubmissionId> taxReturnIdAndSubmissionIds =
                List.of(taxReturnIdAndSubmissionId1, taxReturnIdAndSubmissionId2);

        // Wrap the payload and add a version header since that's what ultimately gets sent
        AbstractPendingSubmissionPayload payload = new PendingSubmissionPayloadV1(taxReturnIdAndSubmissionIds);
        VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> queueMessage =
                new VersionedPendingSubmissionMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        PendingSubmissionMessageVersion.V1.getVersion()));

        sqsConnectionSetupService.sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(
                taxReturnIdAndSubmissionIds);

        ObjectMapper mapper = new ObjectMapper();
        String jsonBody = mapper.writeValueAsString(queueMessage);

        SendMessageRequest sendMessageRequest = SendMessageRequest.builder()
                .queueUrl(pendingSubmissionQueueUrl)
                .messageBody(jsonBody)
                .build();

        ArgumentCaptor<SendMessageRequest> argumentCaptor = ArgumentCaptor.forClass(SendMessageRequest.class);
        verify(sqsClient, times(1)).sendMessage(argumentCaptor.capture());
        assertEquals(sendMessageRequest, argumentCaptor.getValue());
    }

    @Test
    public void sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue_publishDisabled()
            throws JsonProcessingException {
        // Turn off pending submission queue publishing
        when(messageQueueConfig.isPendingSubmissionPublishEnabled()).thenReturn(false);
        sqsConnectionSetupService = new SqsConnectionSetupService(sqsClient, messageQueueConfig);

        sqsConnectionSetupService.sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(List.of());

        // Verify that the message was never sent
        verify(sqsClient, never()).sendMessage(any(SendMessageRequest.class));
    }
}
