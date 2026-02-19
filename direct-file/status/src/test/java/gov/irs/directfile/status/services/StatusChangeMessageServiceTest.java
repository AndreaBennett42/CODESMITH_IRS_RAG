package gov.irs.directfile.status.services;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.PublisherException;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.status.StatusChangeMessageVersion;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;
import gov.irs.directfile.models.message.status.payload.StatusChangePayloadV1;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.doThrow;

@ExtendWith(MockitoExtension.class)
class StatusChangeMessageServiceTest {
    private StatusChangeMessageService statusChangeMessageService;

    @Mock
    private StatusChangeSqsPublisher sqsPublisher;

    @Mock
    private StatusChangeSnsPublisher snsPublisher;

    private static final Map<String, List<String>> v1Object;
    private static final String v1Json;

    static {
        ObjectMapper objectMapper = new ObjectMapper();

        v1Object = Map.of(
                "accepted", List.of("11111111", "33333333"),
                "rejected", List.of("22222222", "44444444"));

        VersionedStatusChangeMessage<AbstractStatusChangePayload> v1VersionedObject =
                new VersionedStatusChangeMessage<>(
                        new StatusChangePayloadV1(v1Object),
                        new QueueMessageHeaders()
                                .addHeader(MessageHeaderAttribute.VERSION, StatusChangeMessageVersion.V1.getVersion()));

        try {
            v1Json = objectMapper.writeValueAsString(v1VersionedObject);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    @BeforeEach
    public void setup() {
        statusChangeMessageService =
                new StatusChangeMessageService(List.of(sqsPublisher, snsPublisher), new ObjectMapper());
    }

    @Test
    public void publishStatusChangePayloadV1_success() {
        statusChangeMessageService.publishStatusChangePayloadV1(v1Object);

        ArgumentCaptor<String> sqsPublishArgumentCaptor = ArgumentCaptor.forClass(String.class);
        verify(sqsPublisher, times(1)).publish(sqsPublishArgumentCaptor.capture());
        assertEquals(v1Json, sqsPublishArgumentCaptor.getValue());

        ArgumentCaptor<String> snsPublishArgumentCaptor = ArgumentCaptor.forClass(String.class);
        verify(snsPublisher, times(1)).publish(snsPublishArgumentCaptor.capture());
        assertEquals(v1Json, snsPublishArgumentCaptor.getValue());
    }

    @ParameterizedTest
    @NullAndEmptySource
    public void publishStatusChangePayloadV1_nullOrEmptyPublishers(List<StatusChangePublisher> publishers) {
        StatusChangeMessageService noPublishersService = new StatusChangeMessageService(publishers, new ObjectMapper());
        noPublishersService.publishStatusChangePayloadV1(v1Object);
        verify(sqsPublisher, never()).publish(any());
        verify(snsPublisher, never()).publish(any());
    }

    @Test
    public void publishSubmissionConfirmationPayloadV1_writeValueAsStringFails() throws JsonProcessingException {
        ObjectMapper mockMapper = Mockito.mock(ObjectMapper.class);
        StatusChangeMessageService testService =
                new StatusChangeMessageService(List.of(sqsPublisher, snsPublisher), mockMapper);
        doThrow(new JsonProcessingException("bad json") {}).when(mockMapper).writeValueAsString(any());

        assertThrows(PublisherException.class, () -> testService.publishStatusChangePayloadV1(v1Object));
    }

    @Test
    public void publishStatusChangePayloadV1_publishFails() {
        doThrow(new PublisherException("could not publish")).when(snsPublisher).publish(any());

        assertThrows(PublisherException.class, () -> statusChangeMessageService.publishStatusChangePayloadV1(v1Object));
    }
}
