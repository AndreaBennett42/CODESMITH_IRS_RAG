package gov.irs.directfile.models.message.dispatch.payload;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME)
@JsonSubTypes({
    @JsonSubTypes.Type(value = DispatchPayloadV1.class, name = "DispatchPayloadV1"),
})
@SuppressWarnings("PMD.AbstractClassWithoutAbstractMethod")
public abstract class AbstractDispatchPayload {}
