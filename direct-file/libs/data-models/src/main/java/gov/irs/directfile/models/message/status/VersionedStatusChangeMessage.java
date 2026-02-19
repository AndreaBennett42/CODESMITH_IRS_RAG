package gov.irs.directfile.models.message.status;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessage;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;

public class VersionedStatusChangeMessage<T extends AbstractStatusChangePayload>
        implements QueueMessage<AbstractStatusChangePayload> {
    private final T payload;
    private final QueueMessageHeaders headers;

    @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
    public VersionedStatusChangeMessage(
            @JsonProperty("payload") T payload, @JsonProperty("headers") QueueMessageHeaders headers) {
        if (headers.getAttribute(MessageHeaderAttribute.VERSION).isEmpty()) {
            throw new IllegalArgumentException(
                    "Unable to instantiate VersionedStatusChangeMessage. Headers must include a version attribute.");
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
