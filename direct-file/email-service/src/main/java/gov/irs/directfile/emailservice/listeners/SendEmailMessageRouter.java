package gov.irs.directfile.emailservice.listeners;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

import gov.irs.directfile.emailservice.listeners.handlers.email.EmailProcessingResults;
import gov.irs.directfile.emailservice.listeners.handlers.email.SendEmailHandler;
import gov.irs.directfile.emailservice.listeners.handlers.email.SendEmailV1Handler;
import gov.irs.directfile.emailservice.listeners.handlers.email.UnsupportedMessageVersionHandler;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.email.SendEmailMessageVersion;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@Service
public class SendEmailMessageRouter {
    private final Map<SendEmailMessageVersion, SendEmailHandler> handlers = new HashMap<>();
    private final UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    public SendEmailMessageRouter(
            UnsupportedMessageVersionHandler unsupportedMessageVersionHandler, SendEmailV1Handler sendEmailV1Handler) {
        this.unsupportedMessageVersionHandler = unsupportedMessageVersionHandler;
        this.handlers.put(SendEmailMessageVersion.V1, sendEmailV1Handler);
    }

    public EmailProcessingResults handleSendEmailMessage(VersionedSendEmailMessage<AbstractSendEmailPayload> message) {
        QueueMessageHeaders headers = message.getHeaders();
        Optional<String> versionOptional = headers.getAttribute(MessageHeaderAttribute.VERSION);

        // We can assume a version number is present here since a VersionedSendEmailMessage
        // mandates that on construction.

        // Get enum for this version (may throw UnsupportedVersionException which we should handle and
        // rethrow to get message back on queue/DLQ).
        SendEmailMessageVersion version;
        try {
            version = SendEmailMessageVersion.getEnum(versionOptional.get());
        } catch (UnsupportedVersionException e) {
            unsupportedMessageVersionHandler.handleSendEmailMessage(message);
            throw e;
        }

        // Get handler.  If not found, we have an unsupported version so handle and throw exception.
        SendEmailHandler handler = handlers.get(version);
        if (handler == null) {
            unsupportedMessageVersionHandler.handleSendEmailMessage(message);
            throw new UnsupportedVersionException(
                    String.format("No handler found for SendEmailMessage (%s)", version.getVersion()));
        }

        // We should have a good handler, so handle the message.
        return handler.handleSendEmailMessage(message);
    }
}
