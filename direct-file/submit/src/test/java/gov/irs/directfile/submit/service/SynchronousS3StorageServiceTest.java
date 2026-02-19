package gov.irs.directfile.submit.service;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.services.s3.model.CommonPrefix;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Response;
import software.amazon.encryption.s3.S3EncryptionClient;

import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.DocumentStoreConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SynchronousS3StorageServiceTest {
    @Mock
    S3EncryptionClient s3EncryptionClient;

    @Mock
    Config config;

    @Mock
    DocumentStoreConfig documentStoreConfig;

    SynchronousS3StorageService s3StorageService;
    String bucketName = "bucketName";

    @BeforeEach
    void setup() {
        when(documentStoreConfig.getEnvironmentPrefix()).thenReturn("environmentPrefix");
        when(config.getDocumentStore()).thenReturn(documentStoreConfig);

        s3StorageService = new SynchronousS3StorageService(s3EncryptionClient, bucketName, config);
    }

    @Test
    public void getSubFolders_givenNonTruncatedResponseFromS3_returnsResults() {
        CommonPrefix commonPrefix1 = CommonPrefix.builder().prefix("prefix1").build();

        CommonPrefix commonPrefix2 = CommonPrefix.builder().prefix("prefix2").build();

        ListObjectsV2Response response = ListObjectsV2Response.builder()
                .isTruncated(false)
                .commonPrefixes(List.of(commonPrefix1, commonPrefix2))
                .build();
        when(s3EncryptionClient.listObjectsV2(any(ListObjectsV2Request.class))).thenReturn(response);

        List<String> result = s3StorageService.getSubFolders("objectKey");

        assertEquals(List.of("prefix1", "prefix2"), result);
        verify(s3EncryptionClient, times(1)).listObjectsV2(any(ListObjectsV2Request.class));
    }

    @Test
    public void getSubFolders_givenTruncatedResponseFromS3_makesAdditionalCallToS3() {
        CommonPrefix commonPrefix1 = CommonPrefix.builder().prefix("prefix1").build();

        CommonPrefix commonPrefix2 = CommonPrefix.builder().prefix("prefix2").build();

        CommonPrefix commonPrefix3 = CommonPrefix.builder().prefix("prefix3").build();

        ListObjectsV2Response firstResponse = ListObjectsV2Response.builder()
                .isTruncated(true)
                .commonPrefixes(List.of(commonPrefix1, commonPrefix2))
                .build();

        ListObjectsV2Response secondResponse = ListObjectsV2Response.builder()
                .isTruncated(false)
                .commonPrefixes(List.of(commonPrefix3))
                .build();

        when(s3EncryptionClient.listObjectsV2(any(ListObjectsV2Request.class)))
                .thenReturn(firstResponse)
                .thenReturn(secondResponse);

        List<String> result = s3StorageService.getSubFolders("objectKey");

        assertEquals(List.of("prefix1", "prefix2", "prefix3"), result);
        verify(s3EncryptionClient, times(2)).listObjectsV2(any(ListObjectsV2Request.class));
    }
}
