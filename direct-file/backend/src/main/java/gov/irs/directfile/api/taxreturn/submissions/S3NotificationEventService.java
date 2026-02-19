package gov.irs.directfile.api.taxreturn.submissions;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;

import gov.irs.directfile.api.config.S3ConfigurationProperties;

@Slf4j
@Service
@EnableConfigurationProperties(S3ConfigurationProperties.class)
public class S3NotificationEventService {

    private final S3Client s3Client;
    private final String environmentPrefix;

    private final S3NotificationEventRouter s3NotificationEventRouter;

    String bucketName;

    ObjectMapper mapper = new ObjectMapper();

    public S3NotificationEventService(
            S3NotificationEventRouter s3NotificationEventRouter,
            @Qualifier("s3WithoutEncryption") S3Client s3Client,
            S3ConfigurationProperties s3ConfigurationProperties) {
        this.s3NotificationEventRouter = s3NotificationEventRouter;
        this.s3Client = s3Client;
        this.bucketName = s3ConfigurationProperties.getS3().getOperationsJobsBucket();
        this.environmentPrefix = s3ConfigurationProperties.getS3().getEnvironmentPrefix();
    }

    public void handleS3NotificationEvent(String rawText) throws JsonProcessingException {
        log.info("Handling S3 notification event and fetching file from S3");
        try {
            Map<String, String> sqsMessage = mapper.readValue(rawText, new TypeReference<>() {});
            String objectKey = sqsMessage.get("path");
            JsonNode s3FileJson = loadObjectFromS3(objectKey, bucketName);
            s3NotificationEventRouter.routeMessage(s3FileJson);
        } catch (Exception e) {
            log.error("Error handling S3 notification event {}", e.getClass(), e);
        }
    }

    public JsonNode loadObjectFromS3(String objectKey, String bucketName) {
        try {
            log.info("Fetching {} object from S3 bucket {}", objectKey, bucketName);
            byte[] s3ObjectBytes = getObjectBytes(objectKey, bucketName);
            return mapper.readTree(s3ObjectBytes);
        } catch (Exception e) {
            log.error("Error fetching {} object from S3: {}", objectKey, e.getMessage());
        }
        return null;
    }

    protected byte[] getObjectBytes(String objectKey, String bucketName) {
        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(bucketName)
                .key(ensureEnvironmentPrefixExists(objectKey))
                .build();
        return s3Client.getObjectAsBytes(getObjectRequest).asByteArray();
    }

    private String ensureEnvironmentPrefixExists(String objectKey) {
        String envAwareKey = StringUtils.prependIfMissing(objectKey, environmentPrefix);
        log.info("Building objectKey {}", envAwareKey);
        return envAwareKey;
    }
}
