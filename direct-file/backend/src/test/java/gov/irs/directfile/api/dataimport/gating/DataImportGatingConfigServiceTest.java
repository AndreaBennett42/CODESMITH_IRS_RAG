package gov.irs.directfile.api.dataimport.gating;

import java.time.Duration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;

import gov.irs.directfile.api.cache.CacheService;
import gov.irs.directfile.api.config.DataImportGatingS3Properties;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class DataImportGatingConfigServiceTest {

    @Mock(name = "s3WithoutEncryption")
    S3Client mockS3Client;

    @Mock
    CacheService mockCacheService;

    private DataImportGatingConfigService dataImportGatingConfigService;

    private String dataImportGatingBucket = "bucketName";
    private String dataImportGatingObject = "objectName";
    private Duration dataImportGatingExpiration = Duration.ofMinutes(1);
    private String environmentPrefix = "environmentPrefix";

    @BeforeEach
    void setup() throws Exception {
        DataImportGatingS3Properties dataImportGatingProperties = new DataImportGatingS3Properties(
                environmentPrefix, dataImportGatingBucket, dataImportGatingObject, dataImportGatingExpiration);
        dataImportGatingConfigService =
                new DataImportGatingConfigService(mockS3Client, dataImportGatingProperties, mockCacheService);
    }

    @Test
    void getDataImportGating_withinAllowList_thenReturnsConfig() {
        // given
        String dataImportGatingConfigString =
                """
				{					
					"percentages": [
						{
							"behavior": 1,
							"percentage": 30
						},
						{
							"behavior": 3,
							"percentage": 70
						}
					],
					"windowing": [
						{
							"start": "2025-01-01T00:00:00Z",
							"end": "2025-01-31T23:59:59Z"
						},
						{
							"start": "2025-02-01T00:00:00Z",
							"end": "2025-02-15T23:59:59Z"
						}	
					]					
				}
				""";

        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, dataImportGatingConfigString.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);

        // when
        DataImportGatingConfig result = dataImportGatingConfigService.getGatingS3Config();

        // then
        assertTrue(result.getPercentages().size() == 2);
        assertTrue(result.getWindowing().size() == 2);
    }

    @Test
    void getDataImportGating_whenExceptionThrown_thenReturnsNull() {
        // given
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenThrow(S3Exception.class);

        // when
        DataImportGatingConfig result = dataImportGatingConfigService.getGatingS3Config();

        // then
        assertTrue(result == null);
    }
}
