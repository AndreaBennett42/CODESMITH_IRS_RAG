package gov.irs.directfile.models.message.email;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;
import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class VersionedSendEmailMessageTest {
    @Test
    public void itCanSerializeAndDeserializeGenericMessage() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> emails = new HashMap<>();
        emails.put(HtmlTemplate.ACCEPTED, List.of(new SendEmailQueueMessageBody()));

        SendEmailPayloadV1 payloadV1 = new SendEmailPayloadV1(emails);

        VersionedSendEmailMessage<AbstractSendEmailPayload> versionedSendEmailMessage = new VersionedSendEmailMessage<>(
                payloadV1,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, SendEmailMessageVersion.V1.getVersion()));

        String rawText = objectMapper.writeValueAsString(versionedSendEmailMessage);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itCanSerializeDeserializePayload() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        AbstractSendEmailPayload payload = new SendEmailPayloadV1(new HashMap<>());

        String rawText = objectMapper.writeValueAsString(payload);
        objectMapper.readValue(rawText, new TypeReference<>() {});
    }

    @Test
    public void itThrowsWhenNoVersionAttributeInHeaders() {
        SendEmailPayloadV1 payloadV1 = new SendEmailPayloadV1(new HashMap<>());
        assertThrows(
                IllegalArgumentException.class,
                () -> new VersionedSendEmailMessage<>(payloadV1, new QueueMessageHeaders()));
    }

    @Test
    public void itCorrectlyParsesVersionIntoEnum() {
        SendEmailMessageVersion v1 = SendEmailMessageVersion.V1;
        String version = SendEmailMessageVersion.V1.getVersion();

        SendEmailMessageVersion versioned = SendEmailMessageVersion.getEnum(version);
        Assertions.assertEquals(v1, versioned);
    }

    @Test
    public void itThrowsExceptionOnBadVersionIntoEnum() {
        String badVersion = "bad";

        assertThrows(UnsupportedVersionException.class, () -> SendEmailMessageVersion.getEnum(badVersion));
    }
}
