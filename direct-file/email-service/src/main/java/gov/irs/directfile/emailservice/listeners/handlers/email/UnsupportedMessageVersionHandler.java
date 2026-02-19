package gov.irs.directfile.emailservice.listeners.handlers.email;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;

@Service
@Slf4j
public class UnsupportedMessageVersionHandler implements SendEmailHandler {
    @Override
    public EmailProcessingResults handleSendEmailMessage(VersionedSendEmailMessage<AbstractSendEmailPayload> payload) {
        log.error("Unable to process Send Email Message. Headers: {} ", payload.getHeaders());
        // Return 0 sent emails so the original message isn't acknowledged and gets put back on queue
        return new EmailProcessingResults(0, 0, 0, null, null);
    }
}
