package gov.irs.directfile.submit.service;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.dispatch.DispatchMessageVersion;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.submit.service.handlers.dispatch.DispatchHandler;
import gov.irs.directfile.submit.service.handlers.dispatch.DispatchV1Handler;
import gov.irs.directfile.submit.service.handlers.dispatch.UnsupportedMessageVersionHandler;

@Service
@SuppressWarnings("PMD.SignatureDeclareThrowsException")
public class DispatchMessageRouter {
    private final Map<DispatchMessageVersion, DispatchHandler> handlers = new HashMap<>();
    private final UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    public DispatchMessageRouter(
            UnsupportedMessageVersionHandler unsupportedMessageVersionHandler, DispatchV1Handler dispatchV1Handler) {
        this.unsupportedMessageVersionHandler = unsupportedMessageVersionHandler;
        this.handlers.put(DispatchMessageVersion.V1, dispatchV1Handler);
    }

    public void handleDispatchMessage(VersionedDispatchMessage<AbstractDispatchPayload> message) throws Exception {
        QueueMessageHeaders headers = message.getHeaders();
        Optional<String> versionOptional = headers.getAttribute(MessageHeaderAttribute.VERSION);

        // We can assume a version number is present here since a VersionedDispatchMessage
        // mandates that on construction.

        // Get enum for this version (may throw UnsupportedVersionException which we should handle and
        // rethrow to get message back on queue/DLQ).
        DispatchMessageVersion version;
        try {
            version = DispatchMessageVersion.getEnum(versionOptional.get());
        } catch (UnsupportedVersionException e) {
            unsupportedMessageVersionHandler.handleDispatchMessage(message);
            throw e;
        }

        // Get handler.  If not found, we have an unsupported version so handle and throw exception.
        DispatchHandler handler = handlers.get(version);
        if (handler == null) {
            unsupportedMessageVersionHandler.handleDispatchMessage(message);
            throw new UnsupportedVersionException(
                    String.format("No handler found for DispatchMessageVersion (%s)", version.getVersion()));
        }

        // We should have a good handler, so handle the message.
        handler.handleDispatchMessage(message);
    }
}
