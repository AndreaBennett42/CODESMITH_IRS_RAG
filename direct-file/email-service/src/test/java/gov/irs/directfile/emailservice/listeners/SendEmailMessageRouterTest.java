package gov.irs.directfile.emailservice.listeners;

import java.util.HashMap;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.emailservice.listeners.handlers.email.SendEmailV1Handler;
import gov.irs.directfile.emailservice.listeners.handlers.email.UnsupportedMessageVersionHandler;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.email.SendEmailMessageVersion;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class SendEmailMessageRouterTest {

    @Mock
    public SendEmailV1Handler sendEmailV1Handler;

    @Mock
    public UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    private SendEmailMessageRouter sendEmailMessageRouter;

    @BeforeEach
    public void setup() {
        sendEmailMessageRouter = new SendEmailMessageRouter(unsupportedMessageVersionHandler, sendEmailV1Handler);
    }

    @Test
    public void itGetsHandlerWhenProvidedAValidSendEmailMessageVersion() {
        // Arrange: Create a VersionedSendEmailMessage, with a header specifying V1
        AbstractSendEmailPayload payload = new SendEmailPayloadV1(new HashMap<>());
        VersionedSendEmailMessage<AbstractSendEmailPayload> queueMessage = new VersionedSendEmailMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, SendEmailMessageVersion.V1.getVersion()));

        // Act: Call handleSendEmailMessage()
        assertDoesNotThrow(() -> {
            sendEmailMessageRouter.handleSendEmailMessage(queueMessage);

            // Assert: Expect that an exception was not thrown, and that the
            // SendEmailV1Handler.handleSendEmailMessage() was called
            verify(sendEmailV1Handler, times(1)).handleSendEmailMessage(any());
        });
    }

    @Test
    public void itHandlesUnsupportedVersions() {
        // Arrange: Create a VersionedSendEmailMessage, with a header specifying an unsupported version
        AbstractSendEmailPayload payload = new SendEmailPayloadV1(new HashMap<>());
        String unsupportedVersion = "9.0.EGG";

        VersionedSendEmailMessage<AbstractSendEmailPayload> messageWithUnsupportedVersion =
                new VersionedSendEmailMessage<>(
                        payload,
                        new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, unsupportedVersion));

        // Act: call handleSendEmailMessage()
        assertThrows(UnsupportedVersionException.class, () -> {
            sendEmailMessageRouter.handleSendEmailMessage(messageWithUnsupportedVersion);

            // Assert: Check that we called the UnsupportedVersionHandler
            verify(unsupportedMessageVersionHandler, times(1)).handleSendEmailMessage(any());
        });
    }
}
