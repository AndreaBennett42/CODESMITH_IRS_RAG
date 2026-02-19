package gov.irs.directfile.models.message.confirmation.payload;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME)
@JsonSubTypes({
    @JsonSubTypes.Type(value = SubmissionConfirmationPayloadV1.class, name = "SubmissionConfirmationPayloadV1"),
    @JsonSubTypes.Type(value = SubmissionConfirmationPayloadV2.class, name = "SubmissionConfirmationPayloadV2"),
})
@SuppressWarnings("PMD.AbstractClassWithoutAbstractMethod")
public abstract class AbstractSubmissionConfirmationPayload {}
