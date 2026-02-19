package gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.SubmissionConfirmationMessageVersion;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class SubmissionConfirmationV2HandlerTest {

    @Mock
    ConfirmationService confirmationService;

    @InjectMocks
    SubmissionConfirmationV2Handler handler;

    @Test
    void
            givenSubmissionConfirmations_whenHandleSubmissionConfirmationMessage_thenConfirmationServiceHandleSubmissionConfirmationsCalled() {
        SubmissionConfirmationPayloadV2Entry entry = new SubmissionConfirmationPayloadV2Entry(
                new TaxReturnSubmissionReceipt(
                        UUID.randomUUID(), UUID.randomUUID().toString(), null, null),
                SubmissionEventTypeEnum.SUBMITTED,
                Map.of());
        SubmissionConfirmationPayloadV2 payloadV2 = new SubmissionConfirmationPayloadV2(List.of(entry));
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message =
                new VersionedSubmissionConfirmationMessage<>(
                        payloadV2,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));
        handler.handleSubmissionConfirmationMessage(message);
        verify(confirmationService, times(1)).handleSubmissionConfirmations(anyList());
        verify(confirmationService, times(0)).handleSubmissionFailures(anyList());
    }

    @Test
    void
            givenFailedConfirmations_whenHandleSubmissionConfirmationMessage_thenConfirmationServiceHandleFailedConfirmationsCalled() {
        SubmissionConfirmationPayloadV2Entry entry = new SubmissionConfirmationPayloadV2Entry(
                new TaxReturnSubmissionReceipt(
                        UUID.randomUUID(), UUID.randomUUID().toString(), null, null),
                SubmissionEventTypeEnum.FAILED,
                null);
        SubmissionConfirmationPayloadV2 payloadV2 = new SubmissionConfirmationPayloadV2(List.of(entry));
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message =
                new VersionedSubmissionConfirmationMessage<>(
                        payloadV2,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));
        handler.handleSubmissionConfirmationMessage(message);
        verify(confirmationService, times(0)).handleSubmissionConfirmations(anyList());
        verify(confirmationService, times(1)).handleSubmissionFailures(anyList());
    }

    @Test
    void
            givenSubmissionAndFailedConfirmations_whenHandleSubmissionConfirmationMessage_thenBothConfirmationServiceHandlersCalled() {
        SubmissionConfirmationPayloadV2Entry entry1 = new SubmissionConfirmationPayloadV2Entry(
                new TaxReturnSubmissionReceipt(
                        UUID.randomUUID(), UUID.randomUUID().toString(), null, null),
                SubmissionEventTypeEnum.SUBMITTED,
                null);
        SubmissionConfirmationPayloadV2Entry entry2 = new SubmissionConfirmationPayloadV2Entry(
                new TaxReturnSubmissionReceipt(
                        UUID.randomUUID(), UUID.randomUUID().toString(), null, null),
                SubmissionEventTypeEnum.FAILED,
                null);
        SubmissionConfirmationPayloadV2 payloadV2 = new SubmissionConfirmationPayloadV2(List.of(entry1, entry2));
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message =
                new VersionedSubmissionConfirmationMessage<>(
                        payloadV2,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));
        handler.handleSubmissionConfirmationMessage(message);
        verify(confirmationService, times(1)).handleSubmissionConfirmations(anyList());
        verify(confirmationService, times(1)).handleSubmissionFailures(anyList());
    }

    @Test
    void givenOtherConfirmations_whenHandleSubmissionConfirmationMessage_thenConfirmationServiceNotCalled() {
        SubmissionConfirmationPayloadV2Entry entry = new SubmissionConfirmationPayloadV2Entry(
                new TaxReturnSubmissionReceipt(
                        UUID.randomUUID(), UUID.randomUUID().toString(), null, null),
                SubmissionEventTypeEnum.ACCEPTED,
                null);
        SubmissionConfirmationPayloadV2 payloadV2 = new SubmissionConfirmationPayloadV2(List.of(entry));
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message =
                new VersionedSubmissionConfirmationMessage<>(
                        payloadV2,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));
        handler.handleSubmissionConfirmationMessage(message);
        verify(confirmationService, times(0)).handleSubmissionConfirmations(anyList());
        verify(confirmationService, times(0)).handleSubmissionFailures(anyList());
    }
}
