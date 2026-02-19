package gov.irs.directfile.api.taxreturn.submissions;

import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;

import gov.irs.directfile.api.config.S3ConfigurationProperties;
import gov.irs.directfile.api.config.S3ConfigurationProperties.S3;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class S3NotificationEventServiceTest {

    S3NotificationEventService s3NotificationEventService;

    S3ConfigurationProperties s3ConfigurationProperties;

    @Mock
    S3NotificationEventRouter s3NotificationEventRouter;

    @Mock(name = "s3WithoutEncryption")
    S3Client mockS3Client;

    String sqsMessage = "{\"path\": \"adhoc_job.json\"}";
    String technicalErrorResolvedJson =
            "{\"key\":\"technical_error_resolved\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\"]}}";

    ObjectMapper mapper = new ObjectMapper();

    @BeforeEach
    public void setup() {
        s3ConfigurationProperties = new S3ConfigurationProperties(
                null, null, new S3("", "", 0, "", "", "some-bucket", "some-operations-jobs-bucket", "dev"));
        s3NotificationEventService =
                new S3NotificationEventService(s3NotificationEventRouter, mockS3Client, s3ConfigurationProperties);
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, technicalErrorResolvedJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
    }

    @Test
    void whenHandleS3NotificationEvent_thenReadsRawText_thenCallsS3NotificationEventRouterrouteMessage()
            throws JsonProcessingException {
        s3NotificationEventService.handleS3NotificationEvent(sqsMessage);
        verify(s3NotificationEventRouter, times(1)).routeMessage(any());
    }

    @Test
    void whenLoadObjectFromS3_thenReturnsJsonNode() throws JsonProcessingException {
        Map<String, String> message = mapper.readValue(sqsMessage, new TypeReference<>() {});
        String objectKey = message.get("path");
        assertEquals(objectKey, "adhoc_job.json");
        var s3FileJson = s3NotificationEventService.loadObjectFromS3(objectKey, "some-bucket");
        assertEquals(s3FileJson.getClass(), ObjectNode.class);
    }
}
