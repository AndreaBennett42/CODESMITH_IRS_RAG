package gov.irs.directfile.models.message.confirmation;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessage;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;

public class VersionedSubmissionConfirmationMessage<T extends AbstractSubmissionConfirmationPayload>
        implements QueueMessage<AbstractSubmissionConfirmationPayload> {
    private final T payload;
    private final QueueMessageHeaders headers;

    @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
    public VersionedSubmissionConfirmationMessage(
            @JsonProperty("payload") T payload, @JsonProperty("headers") QueueMessageHeaders headers) {
        if (headers.getAttribute(MessageHeaderAttribute.VERSION).isEmpty()) {
            throw new IllegalArgumentException(
                    "Unable to instantiate VersionedSubmissionConfirmationMessage. Headers must include a version attribute.");
        }

        this.payload = payload;
        this.headers = headers;
    }

    @Override
    public T getPayload() {
        return payload;
    }

    @Override
    public QueueMessageHeaders getHeaders() {
        return headers;
    }
}
