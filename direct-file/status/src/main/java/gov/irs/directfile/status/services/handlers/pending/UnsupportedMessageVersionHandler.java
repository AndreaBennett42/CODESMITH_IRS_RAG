package gov.irs.directfile.status.services.handlers.pending;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;

@Service
@Slf4j
public class UnsupportedMessageVersionHandler implements PendingSubmissionHandler {
    @Override
    public void handlePendingSubmissionMessage(
            VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> payload) {
        log.error("Unable to process Pending Submission Message. Headers: {} ", payload.getHeaders());
    }
}
