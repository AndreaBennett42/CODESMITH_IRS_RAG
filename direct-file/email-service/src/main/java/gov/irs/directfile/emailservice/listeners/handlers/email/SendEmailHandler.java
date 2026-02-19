package gov.irs.directfile.emailservice.listeners.handlers.email;

import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;

public interface SendEmailHandler {
    EmailProcessingResults handleSendEmailMessage(VersionedSendEmailMessage<AbstractSendEmailPayload> message);
}
