package gov.irs.directfile.api.taxreturn.submissions;

import java.util.HashMap;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.taxreturn.submissions.handlers.status.StatusChangeV1Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.status.UnsupportedMessageVersionHandler;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.models.message.status.StatusChangeMessageVersion;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;
import gov.irs.directfile.models.message.status.payload.StatusChangePayloadV1;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class StatusChangeMessageRouterTest {

    @Mock
    public StatusChangeV1Handler statusChangeV1Handler;

    @Mock
    public UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    private StatusChangeMessageRouter statusChangeMessageRouter;

    @BeforeEach
    public void setup() {
        statusChangeMessageRouter =
                new StatusChangeMessageRouter(unsupportedMessageVersionHandler, statusChangeV1Handler);
    }

    @Test
    public void itGetsHandlerWhenProvidedAValidStatusChangeMessageVersion() {
        // Arrange: Create a VersionedStatusChangeMessage, with a header specifying V1
        AbstractStatusChangePayload payload = new StatusChangePayloadV1(new HashMap<>());
        VersionedStatusChangeMessage<AbstractStatusChangePayload> queueMessage = new VersionedStatusChangeMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, StatusChangeMessageVersion.V1.getVersion()));

        // Act: Call handleStatusChangeMessage()
        assertDoesNotThrow(() -> {
            statusChangeMessageRouter.handleStatusChangeMessage(queueMessage);

            // Assert: Expect that an exception was not thrown, and that the
            // StatusChangeV1Handler.handleStatusChangeMessage() was called
            verify(statusChangeV1Handler, times(1)).handleStatusChangeMessage(any());
        });
    }

    @Test
    public void itHandlesUnsupportedVersions() {
        // Arrange: Create a VersionedStatusChangeMessage, with a header specifying an unsupported version
        AbstractStatusChangePayload payload = new StatusChangePayloadV1(new HashMap<>());
        String unsupportedVersion = "9.0.EGG";

        VersionedStatusChangeMessage<AbstractStatusChangePayload> messageWithUnsupportedVersion =
                new VersionedStatusChangeMessage<>(
                        payload,
                        new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, unsupportedVersion));

        // Act: call handleStatusChangeMessage()
        assertThrows(UnsupportedVersionException.class, () -> {
            statusChangeMessageRouter.handleStatusChangeMessage(messageWithUnsupportedVersion);

            // Assert: Check that we called the UnsupportedVersionHandler
            verify(unsupportedMessageVersionHandler, times(1)).handleStatusChangeMessage(any());
        });
    }
}
