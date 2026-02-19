package gov.irs.directfile.models.message.email.payload;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME)
@JsonSubTypes({
    @JsonSubTypes.Type(value = SendEmailPayloadV1.class, name = "SendEmailPayloadV1"),
})
@SuppressWarnings("PMD.AbstractClassWithoutAbstractMethod")
public abstract class AbstractSendEmailPayload {}
