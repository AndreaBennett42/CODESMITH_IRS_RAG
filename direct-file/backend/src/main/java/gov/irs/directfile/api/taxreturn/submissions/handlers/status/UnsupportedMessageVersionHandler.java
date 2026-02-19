package gov.irs.directfile.api.taxreturn.submissions.handlers.status;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;

@Service("StatusChangeUnsupportedMessageVersionHandler")
@Slf4j
public class UnsupportedMessageVersionHandler implements StatusChangeHandler {
    @Override
    public void handleStatusChangeMessage(VersionedStatusChangeMessage<AbstractStatusChangePayload> payload) {
        log.error("Unable to process Status Change Message. Headers: {} ", payload.getHeaders());
    }
}
