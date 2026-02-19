package gov.irs.directfile.api.taxreturn.submissions;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.submissions.handlers.status.StatusChangeHandler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.status.StatusChangeV1Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.status.UnsupportedMessageVersionHandler;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.models.message.status.StatusChangeMessageVersion;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;

@Service
public class StatusChangeMessageRouter {
    private final Map<StatusChangeMessageVersion, StatusChangeHandler> handlers = new HashMap<>();
    private final UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    public StatusChangeMessageRouter(
            UnsupportedMessageVersionHandler unsupportedMessageVersionHandler,
            StatusChangeV1Handler statusSubmissionV1Handler) {
        this.unsupportedMessageVersionHandler = unsupportedMessageVersionHandler;
        this.handlers.put(StatusChangeMessageVersion.V1, statusSubmissionV1Handler);
    }

    public void handleStatusChangeMessage(VersionedStatusChangeMessage<AbstractStatusChangePayload> message) {
        QueueMessageHeaders headers = message.getHeaders();
        Optional<String> versionOptional = headers.getAttribute(MessageHeaderAttribute.VERSION);

        // We can assume a version number is present here since a VersionedStatusChangeMessage
        // mandates that on construction.

        // Get enum for this version (may throw UnsupportedVersionException which we should handle and
        // rethrow to get message back on queue/DLQ).
        StatusChangeMessageVersion version;
        try {
            version = StatusChangeMessageVersion.getEnum(versionOptional.get());
        } catch (UnsupportedVersionException e) {
            unsupportedMessageVersionHandler.handleStatusChangeMessage(message);
            throw e;
        }

        // Get handler.  If not found, we have an unsupported version so handle and throw exception.
        StatusChangeHandler handler = handlers.get(version);
        if (handler == null) {
            unsupportedMessageVersionHandler.handleStatusChangeMessage(message);
            throw new UnsupportedVersionException(
                    String.format("No handler found for StatusChangeMessageVersion (%s)", version.getVersion()));
        }

        // We should have a good handler, so handle the message.
        handler.handleStatusChangeMessage(message);
    }
}
