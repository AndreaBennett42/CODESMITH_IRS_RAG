package gov.irs.directfile.models.message.status;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;
import gov.irs.directfile.models.message.status.payload.StatusChangePayloadV1;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class VersionedStatusChangeMessageTest {
    @Test
    public void itCanSerializeAndDeserializeGenericMessage() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        Map<String, List<String>> statusSubmissionIdMap = new HashMap<>();
        statusSubmissionIdMap.put("accepted", new ArrayList<>(List.of("123456789")));

        AbstractStatusChangePayload payloadV1 = new StatusChangePayloadV1(statusSubmissionIdMap);

        VersionedStatusChangeMessage<AbstractStatusChangePayload> versionedStatusChangeMessage =
                new VersionedStatusChangeMessage<>(
                        payloadV1,
                        new QueueMessageHeaders()
                                .addHeader(MessageHeaderAttribute.VERSION, StatusChangeMessageVersion.V1.getVersion()));

        String rawText = objectMapper.writeValueAsString(versionedStatusChangeMessage);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itCanSerializeDeserializePayload() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        AbstractStatusChangePayload payload = new StatusChangePayloadV1(new HashMap<>());

        String rawText = objectMapper.writeValueAsString(payload);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itThrowsWhenNoVersionAttributeInHeaders() {
        AbstractStatusChangePayload payloadV1 = new StatusChangePayloadV1(new HashMap<>());
        assertThrows(
                IllegalArgumentException.class,
                () -> new VersionedStatusChangeMessage<>(payloadV1, new QueueMessageHeaders()));
    }

    @Test
    public void itCorrectlyParsesVersionIntoEnum() {
        String version = StatusChangeMessageVersion.V1.getVersion();

        StatusChangeMessageVersion versioned = StatusChangeMessageVersion.getEnum(version);
        Assertions.assertEquals(StatusChangeMessageVersion.V1, versioned);
    }

    @Test
    public void itThrowsExceptionOnBadVersionIntoEnum() {
        String badVersion = "bad";

        assertThrows(UnsupportedVersionException.class, () -> StatusChangeMessageVersion.getEnum(badVersion));
    }
}
