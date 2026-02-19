package gov.irs.directfile.status.services;

import java.util.ArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.models.message.pending.PendingSubmissionMessageVersion;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.models.message.pending.payload.PendingSubmissionPayloadV1;
import gov.irs.directfile.status.services.handlers.pending.PendingSubmissionV1Handler;
import gov.irs.directfile.status.services.handlers.pending.UnsupportedMessageVersionHandler;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class PendingSubmissionMessageRouterTest {

    @Mock
    public PendingSubmissionV1Handler pendingSubmissionV1Handler;

    @Mock
    public UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    private PendingSubmissionMessageRouter pendingSubmissionMessageRouter;

    @BeforeEach
    public void setup() {
        pendingSubmissionMessageRouter =
                new PendingSubmissionMessageRouter(unsupportedMessageVersionHandler, pendingSubmissionV1Handler);
    }

    @Test
    public void itGetsHandlerWhenProvidedAValidPendingSubmissionMessageVersion() {
        // Arrange: Create a VersionedPendingSubmissionMessage, with a header specifying V1
        AbstractPendingSubmissionPayload payload = new PendingSubmissionPayloadV1(new ArrayList<>());
        VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> queueMessage =
                new VersionedPendingSubmissionMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        PendingSubmissionMessageVersion.V1.getVersion()));

        // Act: Call handlePendingSubmissionMessage()
        assertDoesNotThrow(() -> {
            pendingSubmissionMessageRouter.handlePendingSubmissionMessage(queueMessage);

            // Assert: Expect that an exception was not thrown, and that the
            // PendingSubmissionV1Handler.handlePendingSubmissionMessage() was called
            verify(pendingSubmissionV1Handler, times(1)).handlePendingSubmissionMessage(any());
        });
    }

    @Test
    public void itHandlesUnsupportedVersions() {
        // Arrange: Create a VersionedPendingSubmissionMessage, with a header specifying an unsupported version
        AbstractPendingSubmissionPayload payload = new PendingSubmissionPayloadV1(new ArrayList<>());
        String unsupportedVersion = "9.0.EGG";

        VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> messageWithUnsupportedVersion =
                new VersionedPendingSubmissionMessage<>(
                        payload,
                        new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, unsupportedVersion));

        // Act: call handlePendingSubmissionMessage()
        assertThrows(UnsupportedVersionException.class, () -> {
            pendingSubmissionMessageRouter.handlePendingSubmissionMessage(messageWithUnsupportedVersion);

            // Assert: Check that we called the UnsupportedVersionHandler
            verify(unsupportedMessageVersionHandler, times(1)).handlePendingSubmissionMessage(any());
        });
    }
}
