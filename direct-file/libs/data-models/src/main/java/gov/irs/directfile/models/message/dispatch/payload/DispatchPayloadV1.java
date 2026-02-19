package gov.irs.directfile.models.message.dispatch.payload;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;

import gov.irs.directfile.models.Dispatch;

@Getter
public class DispatchPayloadV1 extends AbstractDispatchPayload {
    private final Dispatch dispatch;

    /**
     * To allow Jackson to deserialize JSON into this immutable object,
     * we need to tell Jackson that we want to use the constructor
     * for deserialization, since we don't expose any setters.
     * <p>
     * This is done with the @JsonCreator annotation.
     * We also need to tell Jackson which fields in the json
     * correspond to which fields in the constructor.
     * This is done with the @JsonProperty annotation.
     * <p>
     * Used <a href="https://www.baeldung.com/jackson-deserialize-immutable-objects">this link</a> for reference.
     */
    @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
    public DispatchPayloadV1(@JsonProperty("dispatch") Dispatch dispatch) {
        this.dispatch = dispatch;
    }
}
