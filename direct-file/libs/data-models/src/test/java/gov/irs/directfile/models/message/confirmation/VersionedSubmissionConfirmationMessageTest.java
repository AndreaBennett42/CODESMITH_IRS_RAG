package gov.irs.directfile.models.message.confirmation;

import java.util.List;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class VersionedSubmissionConfirmationMessageTest {

    @Test
    public void itCanSerializeAndDeserializeGenericMessage() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        SubmissionConfirmationPayloadV2 payloadV2 = new SubmissionConfirmationPayloadV2(List.of());

        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload>
                versionedSubmissionConfirmationMessage = new VersionedSubmissionConfirmationMessage<>(
                        payloadV2,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));

        String rawText = objectMapper.writeValueAsString(versionedSubmissionConfirmationMessage);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itCanSerializeDeserializePayload() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV2(List.of());

        String rawText = objectMapper.writeValueAsString(payload);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itThrowsWhenNoVersionAttributeInHeaders() {
        SubmissionConfirmationPayloadV2 payloadV2 = new SubmissionConfirmationPayloadV2(List.of());
        assertThrows(
                IllegalArgumentException.class,
                () -> new VersionedSubmissionConfirmationMessage<>(payloadV2, new QueueMessageHeaders()));
    }

    @Test
    public void itCorrectlyParsesVersionIntoEnum() {
        String version = SubmissionConfirmationMessageVersion.V2.getVersion();

        SubmissionConfirmationMessageVersion versioned = SubmissionConfirmationMessageVersion.getEnum(version);
        Assertions.assertEquals(SubmissionConfirmationMessageVersion.V2, versioned);
    }

    @Test
    public void itThrowsExceptionOnBadVersionIntoEnum() {
        String badVersion = "bad";

        assertThrows(UnsupportedVersionException.class, () -> SubmissionConfirmationMessageVersion.getEnum(badVersion));
    }
}
