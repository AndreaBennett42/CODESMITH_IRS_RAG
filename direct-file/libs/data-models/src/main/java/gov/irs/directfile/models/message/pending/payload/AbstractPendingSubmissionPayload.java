package gov.irs.directfile.models.message.pending.payload;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME)
@JsonSubTypes({
    @JsonSubTypes.Type(value = PendingSubmissionPayloadV1.class, name = "PendingSubmissionPayloadV1"),
})
@SuppressWarnings("PMD.AbstractClassWithoutAbstractMethod")
public abstract class AbstractPendingSubmissionPayload {}
