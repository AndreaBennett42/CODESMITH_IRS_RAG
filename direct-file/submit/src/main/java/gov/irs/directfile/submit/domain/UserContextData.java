package gov.irs.directfile.submit.domain;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;

import gov.irs.directfile.audit.events.TinType;

@Getter
public class UserContextData {
    private final String submissionId;
    private final String userId;
    private final String taxReturnId;
    private final String userTin;
    private final TinType userTinType;
    private final String remoteAddress;
    private final String signDate;

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
    public UserContextData(
            @JsonProperty("submissionId") String submissionId,
            @JsonProperty("userId") String userId,
            @JsonProperty("taxReturnId") String taxReturnId,
            @JsonProperty("userTin") String userTin,
            @JsonProperty("userTinType") TinType userTinType,
            @JsonProperty("remoteAddress") String remoteAddress,
            @JsonProperty("signDate") String localDateTime) {
        this.submissionId = submissionId;
        this.userId = userId;
        this.taxReturnId = taxReturnId;
        this.userTin = userTin;
        this.userTinType = userTinType;
        this.remoteAddress = remoteAddress;
        this.signDate = localDateTime;
    }
}
