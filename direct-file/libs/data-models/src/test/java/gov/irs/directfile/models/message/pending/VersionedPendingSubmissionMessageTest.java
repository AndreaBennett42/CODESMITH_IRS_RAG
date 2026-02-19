package gov.irs.directfile.models.message.pending;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.models.message.pending.payload.PendingSubmissionPayloadV1;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class VersionedPendingSubmissionMessageTest {
    @Test
    public void itCanSerializeAndDeserializeGenericMessage() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        AbstractPendingSubmissionPayload payloadV1 = new PendingSubmissionPayloadV1(new ArrayList<>());

        VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> versionedPendingSubmissionMessage =
                new VersionedPendingSubmissionMessage<>(
                        payloadV1,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        PendingSubmissionMessageVersion.V1.getVersion()));

        String rawText = objectMapper.writeValueAsString(versionedPendingSubmissionMessage);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itCanSerializeDeserializePayload() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        List<TaxReturnIdAndSubmissionId> pendings = new ArrayList<>();
        pendings.add(new TaxReturnIdAndSubmissionId(UUID.randomUUID(), "12345679"));

        AbstractPendingSubmissionPayload payload = new PendingSubmissionPayloadV1(pendings);

        String rawText = objectMapper.writeValueAsString(payload);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itThrowsWhenNoVersionAttributeInHeaders() {
        AbstractPendingSubmissionPayload payloadV1 = new PendingSubmissionPayloadV1(new ArrayList<>());
        assertThrows(
                IllegalArgumentException.class,
                () -> new VersionedPendingSubmissionMessage<>(payloadV1, new QueueMessageHeaders()));
    }

    @Test
    public void itCorrectlyParsesVersionIntoEnum() {
        String version = PendingSubmissionMessageVersion.V1.getVersion();

        PendingSubmissionMessageVersion versioned = PendingSubmissionMessageVersion.getEnum(version);
        Assertions.assertEquals(PendingSubmissionMessageVersion.V1, versioned);
    }

    @Test
    public void itThrowsExceptionOnBadVersionIntoEnum() {
        String badVersion = "bad";

        assertThrows(UnsupportedVersionException.class, () -> PendingSubmissionMessageVersion.getEnum(badVersion));
    }
}
