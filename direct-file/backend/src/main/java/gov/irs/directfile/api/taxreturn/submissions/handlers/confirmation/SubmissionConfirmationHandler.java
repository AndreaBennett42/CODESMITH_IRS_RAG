package gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation;

import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;

public interface SubmissionConfirmationHandler {
    void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message);
}
