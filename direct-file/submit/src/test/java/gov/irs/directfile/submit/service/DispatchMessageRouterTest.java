package gov.irs.directfile.submit.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.models.Dispatch;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.dispatch.DispatchMessageVersion;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.dispatch.payload.DispatchPayloadV1;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.submit.service.handlers.dispatch.DispatchV1Handler;
import gov.irs.directfile.submit.service.handlers.dispatch.UnsupportedMessageVersionHandler;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class DispatchMessageRouterTest {

    @Mock
    public DispatchV1Handler dispatchV1Handler;

    @Mock
    public UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    private DispatchMessageRouter dispatchMessageRouter;

    @BeforeEach
    public void setup() {
        dispatchMessageRouter = new DispatchMessageRouter(unsupportedMessageVersionHandler, dispatchV1Handler);
    }

    @Test
    public void itGetsHandlerWhenProvidedAValidDispatchMessageVersion() {
        // Arrange: Create a VersionedDispatchMessage, with a header specifying V1
        AbstractDispatchPayload payload = new DispatchPayloadV1(new Dispatch());
        VersionedDispatchMessage<AbstractDispatchPayload> queueMessage = new VersionedDispatchMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, DispatchMessageVersion.V1.getVersion()));

        // Act: Call handleDispatchMessage()
        assertDoesNotThrow(() -> {
            dispatchMessageRouter.handleDispatchMessage(queueMessage);

            // Assert: Expect that an exception was not thrown, and that the
            // DispatchV1Handler.handleDispatchMessage() was called
            verify(dispatchV1Handler, times(1)).handleDispatchMessage(any());
        });
    }

    @Test
    public void itHandlesUnsupportedVersions() {
        // Arrange: Create a VersionedDispatchMessage, with a header specifying an unsupported version
        AbstractDispatchPayload payload = new DispatchPayloadV1(new Dispatch());
        String unsupportedVersion = "9.0.EGG";

        VersionedDispatchMessage<AbstractDispatchPayload> messageWithUnsupportedVersion =
                new VersionedDispatchMessage<>(
                        payload,
                        new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, unsupportedVersion));

        // Act: call handleDispatchMessage()
        assertThrows(UnsupportedVersionException.class, () -> {
            dispatchMessageRouter.handleDispatchMessage(messageWithUnsupportedVersion);

            // Assert: Check that we called the UnsupportedVersionHandler
            verify(unsupportedMessageVersionHandler, times(1)).handleDispatchMessage(any());
        });
    }
}
