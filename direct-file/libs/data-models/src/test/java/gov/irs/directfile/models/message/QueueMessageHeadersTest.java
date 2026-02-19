package gov.irs.directfile.models.message;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

public class QueueMessageHeadersTest {
    @Test
    public void itCanSerializeDeserializeHeaders() throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();

        QueueMessageHeaders headers = new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, "1.0");

        String rawText = objectMapper.writeValueAsString(headers);
        QueueMessageHeaders asObject = objectMapper.readValue(rawText, new TypeReference<>() {});
    }
}
