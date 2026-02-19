package gov.irs.directfile.emailservice.listeners.handlers.email;

import java.util.HashMap;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class UnsupportedMessageVersionHandlerTest {
    @Test
    void handleSendEmailMessage_success() throws Exception {
        UnsupportedMessageVersionHandler handler = new UnsupportedMessageVersionHandler();

        AbstractSendEmailPayload payload = new SendEmailPayloadV1(new HashMap<>());
        VersionedSendEmailMessage<AbstractSendEmailPayload> queueMessage = new VersionedSendEmailMessage<>(
                payload, new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, "bad"));

        EmailProcessingResults emailProcessingResults = handler.handleSendEmailMessage(queueMessage);

        assertEquals(0, emailProcessingResults.countToSend());
        assertEquals(0, emailProcessingResults.countSent());
        assertEquals(0, emailProcessingResults.countMessagingException());
        assertTrue(emailProcessingResults.emailsToResend().isEmpty());
    }
}
