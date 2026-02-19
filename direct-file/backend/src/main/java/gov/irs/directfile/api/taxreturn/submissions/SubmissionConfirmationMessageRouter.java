package gov.irs.directfile.api.taxreturn.submissions;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.SubmissionConfirmationHandler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.SubmissionConfirmationV1Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.SubmissionConfirmationV2Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation.UnsupportedMessageVersionHandler;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.SubmissionConfirmationMessageVersion;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@Service
public class SubmissionConfirmationMessageRouter {
    private final Map<SubmissionConfirmationMessageVersion, SubmissionConfirmationHandler> handlers = new HashMap<>();
    private final UnsupportedMessageVersionHandler unsupportedMessageVersionHandler;

    public SubmissionConfirmationMessageRouter(
            UnsupportedMessageVersionHandler unsupportedMessageVersionHandler,
            SubmissionConfirmationV1Handler submissionConfirmationV1Handler,
            SubmissionConfirmationV2Handler submissionConfirmationV2Handler) {
        this.unsupportedMessageVersionHandler = unsupportedMessageVersionHandler;
        this.handlers.put(SubmissionConfirmationMessageVersion.V1, submissionConfirmationV1Handler);
        this.handlers.put(SubmissionConfirmationMessageVersion.V2, submissionConfirmationV2Handler);
    }

    public void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message) {
        QueueMessageHeaders headers = message.getHeaders();
        Optional<String> versionOptional = headers.getAttribute(MessageHeaderAttribute.VERSION);

        // We can assume a version number is present here since a VersionedSubmissionConfirmationMessage
        // mandates that on construction.

        // Get enum for this version (may throw UnsupportedVersionException which we should handle and
        // rethrow to get message back on queue/DLQ).
        SubmissionConfirmationMessageVersion version;
        try {
            version = SubmissionConfirmationMessageVersion.getEnum(versionOptional.get());
        } catch (UnsupportedVersionException e) {
            unsupportedMessageVersionHandler.handleSubmissionConfirmationMessage(message);
            throw e;
        }

        // Get handler.  If not found, we have an unsupported version so handle and throw exception.
        SubmissionConfirmationHandler handler = handlers.get(version);
        if (handler == null) {
            unsupportedMessageVersionHandler.handleSubmissionConfirmationMessage(message);
            throw new UnsupportedVersionException(String.format(
                    "No handler found for SubmissionConfirmationMessageVersion (%s)", version.getVersion()));
        }

        // We should have a good handler, so handle the message.
        handler.handleSubmissionConfirmationMessage(message);
    }
}
