package gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation;

import java.util.List;
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
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV1;

import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class SubmissionConfirmationV1HandlerTest {

    @Mock
    ConfirmationService confirmationService;

    @InjectMocks
    SubmissionConfirmationV1Handler handler;

    @Test
    void
            givenSubmissionConfirmations_whenHandleSubmissionConfirmationMessage_thenConfirmationServiceHandleSubmissionConfirmationsCalled() {
        SubmissionConfirmationPayloadV1 payloadV1 =
                new SubmissionConfirmationPayloadV1(List.of(new TaxReturnSubmissionReceipt(
                        UUID.randomUUID(),
                        UUID.randomUUID().toString(),
                        UUID.randomUUID().toString(),
                        null)));
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message =
                new VersionedSubmissionConfirmationMessage<>(
                        payloadV1,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V1.getVersion()));
        handler.handleSubmissionConfirmationMessage(message);
        verify(confirmationService, times(1)).handleSubmissionConfirmations(anyList());
    }
}
