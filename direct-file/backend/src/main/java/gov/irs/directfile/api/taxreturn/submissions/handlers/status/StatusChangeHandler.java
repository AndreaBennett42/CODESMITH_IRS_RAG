package gov.irs.directfile.api.taxreturn.submissions.handlers.status;

import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;

public interface StatusChangeHandler {
    void handleStatusChangeMessage(VersionedStatusChangeMessage<AbstractStatusChangePayload> message);
}
