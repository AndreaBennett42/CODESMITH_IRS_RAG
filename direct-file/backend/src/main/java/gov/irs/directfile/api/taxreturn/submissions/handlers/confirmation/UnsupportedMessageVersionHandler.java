package gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;

@Service
@Slf4j
public class UnsupportedMessageVersionHandler implements SubmissionConfirmationHandler {
    @Override
    public void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> payload) {
        log.error("Unable to process Submission Confirmation Message. Headers: {} ", payload.getHeaders());
    }
}
