package gov.irs.directfile.submit.service;

import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;

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

import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.PublisherException;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.SubmissionConfirmationMessageVersion;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SubmissionConfirmationMessageServiceTest {
    private SubmissionConfirmationMessageService submissionConfirmationMessageService;

    @Mock
    private SubmissionConfirmationSqsPublisher sqsPublisher;

    @Mock
    private SubmissionConfirmationSnsPublisher snsPublisher;

    private static final List<SubmissionConfirmationPayloadV2Entry> v2Object;
    private static final String v2Json;

    static {
        ObjectMapper objectMapper = new ObjectMapper();

        v2Object = List.of(
                new SubmissionConfirmationPayloadV2Entry(
                        new TaxReturnSubmissionReceipt(UUID.randomUUID(), "11111111", "22222222", new Date()),
                        SubmissionEventTypeEnum.SUBMITTED,
                        Map.of("key1", "value1")),
                new SubmissionConfirmationPayloadV2Entry(
                        new TaxReturnSubmissionReceipt(UUID.randomUUID(), "33333333", "44444444", new Date()),
                        SubmissionEventTypeEnum.SUBMITTED,
                        Map.of("key2", "value2")));

        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> v2VersionedObject =
                new VersionedSubmissionConfirmationMessage<>(
                        new SubmissionConfirmationPayloadV2(v2Object),
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));

        try {
            v2Json = objectMapper.writeValueAsString(v2VersionedObject);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    @BeforeEach
    public void setup() {
        submissionConfirmationMessageService =
                new SubmissionConfirmationMessageService(List.of(sqsPublisher, snsPublisher), new ObjectMapper());
    }

    @Test
    public void publishSubmissionConfirmationPayloadV2_success() {
        submissionConfirmationMessageService.publishSubmissionConfirmationPayloadV2(v2Object);

        ArgumentCaptor<String> sqsPublishArgumentCaptor = ArgumentCaptor.forClass(String.class);
        verify(sqsPublisher, times(1)).publish(sqsPublishArgumentCaptor.capture());
        assertEquals(v2Json, sqsPublishArgumentCaptor.getValue());

        ArgumentCaptor<String> snsPublishArgumentCaptor = ArgumentCaptor.forClass(String.class);
        verify(snsPublisher, times(1)).publish(snsPublishArgumentCaptor.capture());
        assertEquals(v2Json, snsPublishArgumentCaptor.getValue());
    }

    @ParameterizedTest
    @NullAndEmptySource
    public void publishSubmissionConfirmationPayloadV2_nullOrEmptyPublishers(
            List<SubmissionConfirmationPublisher> publishers) {
        SubmissionConfirmationMessageService noPublishersService =
                new SubmissionConfirmationMessageService(publishers, new ObjectMapper());
        noPublishersService.publishSubmissionConfirmationPayloadV2(v2Object);
        verify(sqsPublisher, never()).publish(any());
        verify(snsPublisher, never()).publish(any());
    }

    @Test
    public void publishSubmissionConfirmationPayloadV2_writeValueAsStringFails() throws JsonProcessingException {
        ObjectMapper mockMapper = Mockito.mock(ObjectMapper.class);
        SubmissionConfirmationMessageService testService =
                new SubmissionConfirmationMessageService(List.of(sqsPublisher, snsPublisher), mockMapper);
        doThrow(new JsonProcessingException("bad json") {}).when(mockMapper).writeValueAsString(any());

        assertThrows(PublisherException.class, () -> testService.publishSubmissionConfirmationPayloadV2(v2Object));
    }

    @Test
    public void publishSubmissionConfirmationPayloadV2_publishFails() {
        doThrow(new PublisherException("could not publish")).when(snsPublisher).publish(any());

        assertThrows(
                PublisherException.class,
                () -> submissionConfirmationMessageService.publishSubmissionConfirmationPayloadV2(v2Object));
    }
}
