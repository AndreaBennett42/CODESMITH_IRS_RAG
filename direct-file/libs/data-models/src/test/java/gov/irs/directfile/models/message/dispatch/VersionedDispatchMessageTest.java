package gov.irs.directfile.models.message.dispatch;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.Dispatch;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.dispatch.payload.DispatchPayloadV1;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class VersionedDispatchMessageTest {
    @Test
    public void itCanSerializeAndDeserializeGenericMessage() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        AbstractDispatchPayload payloadV1 = new DispatchPayloadV1(new Dispatch());

        VersionedDispatchMessage<AbstractDispatchPayload> versionedDispatchMessage = new VersionedDispatchMessage<>(
                payloadV1,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, DispatchMessageVersion.V1.getVersion()));

        String rawText = objectMapper.writeValueAsString(versionedDispatchMessage);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itCanSerializeDeserializePayload() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        AbstractDispatchPayload payload = new DispatchPayloadV1(Dispatch.testObjectFactory());

        String rawText = objectMapper.writeValueAsString(payload);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itThrowsWhenNoVersionAttributeInHeaders() {
        AbstractDispatchPayload payloadV1 = new DispatchPayloadV1(new Dispatch());
        assertThrows(
                IllegalArgumentException.class,
                () -> new VersionedDispatchMessage<>(payloadV1, new QueueMessageHeaders()));
    }

    @Test
    public void itCorrectlyParsesVersionIntoEnum() {
        String version = DispatchMessageVersion.V1.getVersion();

        DispatchMessageVersion versioned = DispatchMessageVersion.getEnum(version);
        Assertions.assertEquals(DispatchMessageVersion.V1, versioned);
    }

    @Test
    public void itThrowsExceptionOnBadVersionIntoEnum() {
        String badVersion = "bad";

        assertThrows(UnsupportedVersionException.class, () -> DispatchMessageVersion.getEnum(badVersion));
    }
}
