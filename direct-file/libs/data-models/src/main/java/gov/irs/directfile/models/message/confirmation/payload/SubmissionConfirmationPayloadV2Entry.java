package gov.irs.directfile.models.message.confirmation.payload;

import java.util.Map;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;

import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Getter
public class SubmissionConfirmationPayloadV2Entry {
    private final TaxReturnSubmissionReceipt taxReturnSubmissionReceipt;
    private final SubmissionEventTypeEnum eventType;
    private final Map<String, String> metadata;

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
    public SubmissionConfirmationPayloadV2Entry(
            @JsonProperty("taxReturnSubmissionReceipt") TaxReturnSubmissionReceipt taxReturnSubmissionReceipt,
            @JsonProperty("eventType") SubmissionEventTypeEnum eventType,
            @JsonProperty("metadata") Map<String, String> metadata) {
        this.taxReturnSubmissionReceipt = taxReturnSubmissionReceipt;
        this.eventType = eventType;
        this.metadata = metadata;
    }
}
