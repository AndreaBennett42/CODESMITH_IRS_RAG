package gov.irs.directfile.status.services.handlers.pending;

import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;

public interface PendingSubmissionHandler {
    void handlePendingSubmissionMessage(VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> message);
}
