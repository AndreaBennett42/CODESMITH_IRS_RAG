package gov.irs.directfile.models.message.status.payload;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME)
@JsonSubTypes({
    @JsonSubTypes.Type(value = StatusChangePayloadV1.class, name = "StatusChangePayloadV1"),
})
@SuppressWarnings("PMD.AbstractClassWithoutAbstractMethod")
public abstract class AbstractStatusChangePayload {}
