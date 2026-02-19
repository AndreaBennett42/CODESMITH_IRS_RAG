package gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation;

import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV1;

/**
 * @deprecated V1 messages are no longer sent, but keeping handlers until we are sure queues are V1-free
 */
@Deprecated
@Service
public class SubmissionConfirmationV1Handler implements SubmissionConfirmationHandler {

    private final ConfirmationService confirmationService;

    public SubmissionConfirmationV1Handler(ConfirmationService confirmationService) {
        this.confirmationService = confirmationService;
    }

    @Override
    public void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message) {
        SubmissionConfirmationPayloadV1 payload = (SubmissionConfirmationPayloadV1) message.getPayload();
        confirmationService.handleSubmissionConfirmations(payload.getReceipts());
    }
}
