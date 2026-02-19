package gov.irs.directfile.api.taxreturn.submissions;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.SubmissionConfirmationV1Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.SubmissionConfirmationV2Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.UnsupportedMessageVersionHandler;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.SubmissionConfirmationMessageVersion;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV1;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class SubmissionConfirmationMessageRouterTest {

    @Mock
    public SubmissionConfirmationV1Handler submissionConfirmationV1Handler;

    @Mock
    public SubmissionConfirmationV2Handler submissionConfirmationV2Handler;

    @Mock
    public UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    private SubmissionConfirmationMessageRouter submissionConfirmationMessageRouter;

    @BeforeEach
    public void setup() {
        submissionConfirmationMessageRouter = new SubmissionConfirmationMessageRouter(
                unsupportedMessageVersionHandler, submissionConfirmationV1Handler, submissionConfirmationV2Handler);
    }

    @Test
    public void itGetsHandlerWhenProvidedAValidConfirmationMessageVersion() {
        // Arrange: Create a VersionedSubmissionConfirmationMessage, with a header specifying V1
        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV1(List.of());
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> queueMessage =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V1.getVersion()));

        // Act: Call handleSubmissionConfirmationMessage()
        assertDoesNotThrow(() -> {
            submissionConfirmationMessageRouter.handleSubmissionConfirmationMessage(queueMessage);

            // Assert: Expect that an exception was not thrown, and that the
            // SubmissionConfirmationV1Handler.handleSubmissionConfirmationMessage() was called
            verify(submissionConfirmationV1Handler, times(1)).handleSubmissionConfirmationMessage(any());
        });
    }

    @Test
    public void givenConfirmationMessageV2_whenHandleSubmissionConfirmationMessage_thenCorrectHandlerChosen() {
        // Arrange: Create a VersionedSubmissionConfirmationMessage, with a header specifying V2
        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV2(List.of());
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> queueMessage =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));

        // Act: Call handleSubmissionConfirmationMessage()
        assertDoesNotThrow(() -> {
            submissionConfirmationMessageRouter.handleSubmissionConfirmationMessage(queueMessage);

            // Assert: Expect that an exception was not thrown, and that the
            // SubmissionConfirmationV2Handler.handleSubmissionConfirmationMessage() was called
            verify(submissionConfirmationV2Handler, times(1)).handleSubmissionConfirmationMessage(any());
        });
    }

    @Test
    public void itHandlesUnsupportedVersions() {
        // Arrange: Create a VersionedSubmissionMessage, with a header specifying an unsupported version
        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV2(List.of());
        String unsupportedVersion = "9.0.EGG";

        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> messageWithUnsupportedVersion =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, unsupportedVersion));

        // Act: call handleSubmissionConfirmationMessage()
        assertThrows(UnsupportedVersionException.class, () -> {
            submissionConfirmationMessageRouter.handleSubmissionConfirmationMessage(messageWithUnsupportedVersion);

            // Assert: Check that we called the UnsupportedVersionHandler
            verify(unsupportedMessageVersionHandler, times(1)).handleSubmissionConfirmationMessage(any());
        });
    }
}
