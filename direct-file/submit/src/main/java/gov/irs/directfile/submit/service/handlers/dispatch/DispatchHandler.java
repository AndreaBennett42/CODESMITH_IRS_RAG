package gov.irs.directfile.submit.service.handlers.dispatch;

import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;

@SuppressWarnings("PMD.SignatureDeclareThrowsException")
public interface DispatchHandler {
    void handleDispatchMessage(VersionedDispatchMessage<AbstractDispatchPayload> message) throws Exception;
}
