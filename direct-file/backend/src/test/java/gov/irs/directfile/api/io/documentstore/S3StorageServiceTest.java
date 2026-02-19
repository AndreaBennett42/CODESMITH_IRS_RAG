package gov.irs.directfile.api.io.documentstore;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.encryption.s3.S3EncryptionClient;

import gov.irs.directfile.api.config.S3ConfigurationProperties;
import gov.irs.directfile.api.config.S3ConfigurationProperties.S3;
import gov.irs.directfile.api.errors.TaxReturnNotFoundResponseStatusException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class S3StorageServiceTest {

    private final String bucketName = "test-bucket";
    private final String operationsJobsBucketName = "test-operations-jobs-bucket";
    private final String environmentPrefix = "dev/";

    @Mock
    private S3EncryptionClient s3MockClient;

    S3ConfigurationProperties s3ConfigurationProperties = new S3ConfigurationProperties(
            null, null, new S3(null, null, 0, null, null, bucketName, operationsJobsBucketName, environmentPrefix));

    @Test
    @SneakyThrows
    void s3_sends_reads_file_request() throws IOException {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String testUUID = UUID.randomUUID().toString();
        String objectKey = testUUID + ".txt";

        // Test and capture
        try {
            s3StorageService.download(objectKey);
        } catch (IOException e) {
            // We can ignore this error for the test since we just want to capture and compare the s3 request.
            // Because the client is mocked we don't receive an exception that the object with the key we just made up
            // doesn't exist (because the document store doesn't exist here).
            if (!e.getMessage().equals("Empty file found!")) throw e;
        }
        ArgumentCaptor<GetObjectRequest> captor = ArgumentCaptor.forClass(GetObjectRequest.class);
        verify(s3MockClient).getObjectAsBytes(captor.capture());

        // Get value
        GetObjectRequest objectRequest = captor.getValue();

        // Make assertions
        assertEquals(bucketName, objectRequest.bucket());
        assertEquals(environmentPrefix + objectKey, objectRequest.key());
    }

    @Test
    @SneakyThrows
    void s3_sends_reads_file_request_with_environment_prefix() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String testUUID = UUID.randomUUID().toString();
        String objectKey = testUUID + ".txt";

        // Test and capture
        try {
            s3StorageService.download(objectKey);
        } catch (IOException e) {
            // We can ignore this error for the test since we just want to capture and compare the s3 request.
            // Because the client is mocked we don't receive an exception that the object with the key we just made up
            // doesn't exist (because the document store doesn't exist here).
            if (!e.getMessage().equals("Empty file found!")) throw e;
        }
        ArgumentCaptor<GetObjectRequest> captor = ArgumentCaptor.forClass(GetObjectRequest.class);
        verify(s3MockClient).getObjectAsBytes(captor.capture());

        // Get value
        GetObjectRequest objectRequest = captor.getValue();

        // Make assertions
        assertEquals(bucketName, objectRequest.bucket());
        assertEquals(environmentPrefix + objectKey, objectRequest.key());
    }

    @Test
    @SneakyThrows
    void s3_sends_write_file_request() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String testUUID = UUID.randomUUID().toString();
        String objectKey = testUUID + ".txt";
        String originalContent = "expected back";
        InputStream stream = new ByteArrayInputStream(originalContent.getBytes());

        // Test and capture
        s3StorageService.write(objectKey, stream);
        ArgumentCaptor<PutObjectRequest> requestCaptor = ArgumentCaptor.forClass(PutObjectRequest.class);
        ArgumentCaptor<RequestBody> requestBodyCaptor = ArgumentCaptor.forClass(RequestBody.class);
        verify(s3MockClient).putObject(requestCaptor.capture(), requestBodyCaptor.capture());

        // Get test values
        PutObjectRequest objectRequest = requestCaptor.getValue();
        RequestBody bodyRequest = requestBodyCaptor.getValue();
        InputStream capturedStream = bodyRequest.contentStreamProvider().newStream();
        String returnedContents = new String(capturedStream.readAllBytes(), StandardCharsets.UTF_8);

        // Make assertions
        assertEquals(bucketName, objectRequest.bucket());
        assertEquals(environmentPrefix + objectKey, objectRequest.key());
        assertEquals("application/octet-stream", bodyRequest.contentType());
        assertEquals(originalContent, returnedContents);
    }

    @Test
    @SneakyThrows
    void s3_sends_write_file_request_with_environment_prefix() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String testUUID = UUID.randomUUID().toString();
        String objectKey = testUUID + ".txt";
        String originalContent = "expected back";
        InputStream stream = new ByteArrayInputStream(originalContent.getBytes());

        // Test and capture
        s3StorageService.write(objectKey, stream);
        ArgumentCaptor<PutObjectRequest> requestCaptor = ArgumentCaptor.forClass(PutObjectRequest.class);
        ArgumentCaptor<RequestBody> requestBodyCaptor = ArgumentCaptor.forClass(RequestBody.class);
        verify(s3MockClient).putObject(requestCaptor.capture(), requestBodyCaptor.capture());

        // Get test values
        PutObjectRequest objectRequest = requestCaptor.getValue();
        RequestBody bodyRequest = requestBodyCaptor.getValue();
        InputStream capturedStream = bodyRequest.contentStreamProvider().newStream();
        String returnedContents = new String(capturedStream.readAllBytes(), StandardCharsets.UTF_8);

        // Make assertions
        assertEquals(bucketName, objectRequest.bucket());
        assertEquals(environmentPrefix + objectKey, objectRequest.key());
        assertEquals(bodyRequest.contentType(), "application/octet-stream");
        assertEquals(returnedContents, originalContent);
    }

    @Test
    void s3_sends_request_to_lists_objects_at_location() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String baseLocation = "2023/taxreturns/";

        // Test and Capture
        try {
            s3StorageService.listLocationResources(baseLocation);
        } catch (Exception e) {
            // We can ignore this error for the test since we just want to capture and compare the s3 request.
            // Because the client is mocked we don't receive an exception that the object with the key we just made up

        }
        ArgumentCaptor<ListObjectsRequest> requestCaptor = ArgumentCaptor.forClass(ListObjectsRequest.class);
        verify(s3MockClient).listObjects(requestCaptor.capture());

        // Get test values
        ListObjectsRequest objectRequest = requestCaptor.getValue();

        // Make assertions
        assertEquals(bucketName, objectRequest.bucket());
        assertEquals(environmentPrefix + baseLocation, objectRequest.prefix());
    }

    @Test
    void s3_sends_request_to_lists_objects_at_location_with_environment_prefix() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String baseLocation = "2023/taxreturns/";

        // Test and Capture
        try {
            s3StorageService.listLocationResources(baseLocation);
        } catch (Exception e) {
            // We can ignore this error for the test since we just want to capture and compare the s3 request.
            // Because the client is mocked we don't receive an exception that the object with the key we just made up

        }
        ArgumentCaptor<ListObjectsRequest> requestCaptor = ArgumentCaptor.forClass(ListObjectsRequest.class);
        verify(s3MockClient).listObjects(requestCaptor.capture());

        // Get test values
        ListObjectsRequest objectRequest = requestCaptor.getValue();

        // Make assertions
        assertEquals(objectRequest.bucket(), bucketName);
        assertEquals(objectRequest.prefix(), environmentPrefix + baseLocation);
    }

    @Test
    void given_listLocationResources_whenNoResources() {
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        // given
        String baseLocation = "test/";
        String prefixedBaseLocation = environmentPrefix + baseLocation;
        ListObjectsRequest listReq = ListObjectsRequest.builder()
                .bucket(bucketName)
                .prefix(prefixedBaseLocation)
                .build();
        ListObjectsResponse res = ListObjectsResponse.builder().build();
        when(s3MockClient.listObjects(listReq)).thenReturn(res);

        // when
        List<DocumentStoreResource> resources = null;
        try {
            resources = s3StorageService.listLocationResources(prefixedBaseLocation);
        } catch (TaxReturnNotFoundResponseStatusException e) {
            fail();
            return;
        }

        // then
        assertEquals(0, resources.size());
    }

    @Test
    void given_listLocationResources_whenNoResources_with_environment_prefix() {
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        // given
        String baseLocation = "test/";
        String prefixedBaseLocation = environmentPrefix + baseLocation;
        ListObjectsRequest listReq = ListObjectsRequest.builder()
                .bucket(bucketName)
                .prefix(prefixedBaseLocation)
                .build();
        ListObjectsResponse res = ListObjectsResponse.builder().build();
        when(s3MockClient.listObjects(listReq)).thenReturn(res);

        // when
        List<DocumentStoreResource> resources = null;
        try {
            resources = s3StorageService.listLocationResources(prefixedBaseLocation);
        } catch (TaxReturnNotFoundResponseStatusException e) {
            fail();
            return;
        }

        // then
        assertEquals(0, resources.size());
    }

    @Test
    void given_listLocationResources_whenResources() {
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        // given
        String baseLocation = "test/";
        String prefixedBaseLocation = environmentPrefix + baseLocation;
        ListObjectsRequest listReq = ListObjectsRequest.builder()
                .bucket(bucketName)
                .prefix(prefixedBaseLocation)
                .build();
        ListObjectsResponse res = ListObjectsResponse.builder()
                .contents(S3Object.builder()
                        .key(prefixedBaseLocation + "test.txt")
                        .lastModified(mock(Instant.class))
                        .build())
                .build();
        when(s3MockClient.listObjects(listReq)).thenReturn(res);

        // when
        List<DocumentStoreResource> resources = null;
        try {
            resources = s3StorageService.listLocationResources(prefixedBaseLocation);
        } catch (TaxReturnNotFoundResponseStatusException e) {
            throw new RuntimeException(e);
        }

        // then
        assertEquals(1, resources.size());
        assertEquals("test", resources.get(0).getResourceId());
    }

    @Test
    void given_listLocationResources_whenResources_with_environment_prefix() {

        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        // given
        String baseLocation = "test/";
        String prefixedBaseLocation = environmentPrefix + baseLocation;
        ListObjectsRequest listReq = ListObjectsRequest.builder()
                .bucket(bucketName)
                .prefix(prefixedBaseLocation)
                .build();
        ListObjectsResponse res = ListObjectsResponse.builder()
                .contents(S3Object.builder()
                        .key(prefixedBaseLocation + "test.txt")
                        .lastModified(mock(Instant.class))
                        .build())
                .build();
        when(s3MockClient.listObjects(listReq)).thenReturn(res);

        // when
        List<DocumentStoreResource> resources = null;
        try {
            resources = s3StorageService.listLocationResources(prefixedBaseLocation);
        } catch (TaxReturnNotFoundResponseStatusException e) {
            throw new RuntimeException(e);
        }

        // then
        assertEquals(1, resources.size());
        assertEquals("test", resources.get(0).getResourceId());
    }

    @Test
    void s3_sends_copy_request() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String sourceKey = UUID.randomUUID() + ".txt";
        String destinationKey = UUID.randomUUID() + ".txt";

        // Test and capture
        s3StorageService.copy(sourceKey, destinationKey);
        ArgumentCaptor<CopyObjectRequest> requestCaptor = ArgumentCaptor.forClass(CopyObjectRequest.class);
        verify(s3MockClient).copyObject(requestCaptor.capture());

        // Get test values
        CopyObjectRequest objectRequest = requestCaptor.getValue();

        // Make assertions
        assertEquals(bucketName, objectRequest.sourceBucket());
        assertEquals(environmentPrefix + sourceKey, objectRequest.sourceKey());
        assertEquals(bucketName, objectRequest.destinationBucket());
        assertEquals(environmentPrefix + destinationKey, objectRequest.destinationKey());
        assertEquals(MetadataDirective.COPY, objectRequest.metadataDirective());
    }

    @Test
    void s3_sends_copy_request_with_added_metadata() {
        // Setup
        S3StorageService s3StorageService = new S3StorageService(s3MockClient, s3ConfigurationProperties);
        String sourceKey = UUID.randomUUID() + ".txt";
        String destinationKey = UUID.randomUUID() + ".txt";
        Map<String, String> sourceMetadata = Map.of("key1", "value1");
        Map<String, String> addedMetadata = Map.of("key2", "value2");
        Map<String, String> combinedMetadata = new HashMap<>();
        combinedMetadata.putAll(sourceMetadata);
        combinedMetadata.putAll(addedMetadata);

        HeadObjectResponse headResponse = mock(HeadObjectResponse.class);
        when(headResponse.metadata()).thenReturn(sourceMetadata);
        when(s3MockClient.headObject(any(HeadObjectRequest.class))).thenReturn(headResponse);

        // Test and capture
        s3StorageService.copyWithAddedMetadata(sourceKey, destinationKey, addedMetadata);
        ArgumentCaptor<HeadObjectRequest> headRequestCaptor = ArgumentCaptor.forClass(HeadObjectRequest.class);
        ArgumentCaptor<CopyObjectRequest> copyRequestCaptor = ArgumentCaptor.forClass(CopyObjectRequest.class);
        verify(s3MockClient).headObject(headRequestCaptor.capture());
        verify(s3MockClient).copyObject(copyRequestCaptor.capture());

        // Get test values
        HeadObjectRequest objectHeadRequest = headRequestCaptor.getValue();
        CopyObjectRequest objectCopyRequest = copyRequestCaptor.getValue();

        // Make assertions
        assertEquals(bucketName, objectHeadRequest.bucket());
        assertEquals(environmentPrefix + sourceKey, objectHeadRequest.key());
        assertEquals(bucketName, objectCopyRequest.sourceBucket());
        assertEquals(environmentPrefix + sourceKey, objectCopyRequest.sourceKey());
        assertEquals(bucketName, objectCopyRequest.destinationBucket());
        assertEquals(environmentPrefix + destinationKey, objectCopyRequest.destinationKey());
        assertEquals(MetadataDirective.REPLACE, objectCopyRequest.metadataDirective());
        assertEquals(combinedMetadata, objectCopyRequest.metadata());
    }
}
