package gov.irs.directfile.models.message;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonProperty;

public class QueueMessageHeaders {

    @JsonProperty("headers")
    private final Map<MessageHeaderAttribute, String> headers;

    public QueueMessageHeaders() {
        this.headers = new HashMap<>();
    }

    public Optional<String> getAttribute(MessageHeaderAttribute attribute) {
        return Optional.ofNullable(headers.get(attribute));
    }

    public QueueMessageHeaders addHeader(MessageHeaderAttribute attribute, String value) {
        headers.put(attribute, value);
        return this;
    }

    @Override
    public String toString() {
        return "QueueMessageHeaders{" + "headers=" + headers + '}';
    }
}
