package gov.irs.directfile.status.services;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.models.message.pending.PendingSubmissionMessageVersion;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.status.services.handlers.pending.PendingSubmissionHandler;
import gov.irs.directfile.status.services.handlers.pending.PendingSubmissionV1Handler;
import gov.irs.directfile.status.services.handlers.pending.UnsupportedMessageVersionHandler;

@Service
public class PendingSubmissionMessageRouter {
    private final Map<PendingSubmissionMessageVersion, PendingSubmissionHandler> handlers = new HashMap<>();
    private final UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    public PendingSubmissionMessageRouter(
            UnsupportedMessageVersionHandler unsupportedMessageVersionHandler,
            PendingSubmissionV1Handler pendingSubmissionV1Handler) {
        this.unsupportedMessageVersionHandler = unsupportedMessageVersionHandler;
        this.handlers.put(PendingSubmissionMessageVersion.V1, pendingSubmissionV1Handler);
    }

    public void handlePendingSubmissionMessage(
            VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> message) {
        QueueMessageHeaders headers = message.getHeaders();
        Optional<String> versionOptional = headers.getAttribute(MessageHeaderAttribute.VERSION);

        // We can assume a version number is present here since a VersionedPendingSubmissionMessage
        // mandates that on construction.

        // Get enum for this version (may throw UnsupportedVersionException which we should handle and
        // rethrow to get message back on queue/DLQ).
        PendingSubmissionMessageVersion version;
        try {
            version = PendingSubmissionMessageVersion.getEnum(versionOptional.get());
        } catch (UnsupportedVersionException e) {
            unsupportedMessageVersionHandler.handlePendingSubmissionMessage(message);
            throw e;
        }

        // Get handler.  If not found, we have an unsupported version so handle and throw exception.
        PendingSubmissionHandler handler = handlers.get(version);
        if (handler == null) {
            unsupportedMessageVersionHandler.handlePendingSubmissionMessage(message);
            throw new UnsupportedVersionException(
                    String.format("No handler found for PendingSubmissionMessageVersion (%s)", version.getVersion()));
        }

        // We should have a good handler, so handle the message.
        handler.handlePendingSubmissionMessage(message);
    }
}
