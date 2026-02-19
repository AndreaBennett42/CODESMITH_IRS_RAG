package gov.irs.directfile.api.io.documentstore;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.model.CopyObjectRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.HeadObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectResponse;
import software.amazon.awssdk.services.s3.model.ListObjectsRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsResponse;
import software.amazon.awssdk.services.s3.model.MetadataDirective;
import software.amazon.awssdk.services.s3.model.NoSuchBucketException;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.S3Exception;
import software.amazon.awssdk.services.s3.model.S3Object;
import software.amazon.encryption.s3.S3EncryptionClient;
import software.amazon.encryption.s3.S3EncryptionClientException;

import gov.irs.directfile.api.config.S3ConfigurationProperties;
import gov.irs.directfile.api.errors.TaxReturnNotFoundResponseStatusException;

@Service
@Slf4j
@EnableConfigurationProperties(S3ConfigurationProperties.class)
public class S3StorageService {

    private final S3EncryptionClient s3Client;
    private final String bucketName;
    private final String environmentPrefix;

    public S3StorageService(S3EncryptionClient s3, S3ConfigurationProperties s3ConfigurationProperties) {
        this.s3Client = s3;
        this.bucketName = s3ConfigurationProperties.getS3().getBucket();
        this.environmentPrefix = s3ConfigurationProperties.getS3().getEnvironmentPrefix();
    }

    public void write(String objectKeyInput, InputStream payloadStream) throws IOException {
        write(objectKeyInput, payloadStream, null);
    }

    public void write(String objectKeyInput, InputStream payloadStream, Map<String, String> metadata)
            throws IOException {
        String objectKey = ensureEnvironmentPrefixExists(objectKeyInput);
        PutObjectRequest s3ObjReq = PutObjectRequest.builder()
                .bucket(bucketName)
                .key(objectKey)
                .metadata(metadata)
                .build();
        byte[] bytes = IOUtils.toByteArray(payloadStream);

        try {
            s3Client.putObject(s3ObjReq, RequestBody.fromBytes(bytes));
            log.info("Successfully placed " + objectKey + " into bucket " + bucketName);
        } catch (NoSuchBucketException e) {
            throw new IOException(String.format("Bucket %s does not exist or access is denied", bucketName), e);
        } catch (S3Exception e) {
            log.error("Error writing to S3: " + e.getMessage());
            throw new IOException(String.format("Failed to write to bucket %s ", bucketName), e);
        }
    }

    public void copy(String sourceKey, String destinationKey) {
        String sourceKeyWithEnvironmentPrefix = ensureEnvironmentPrefixExists(sourceKey);
        String destinationKeyWithEnvironmentPrefix = ensureEnvironmentPrefixExists(destinationKey);
        CopyObjectRequest copyReq = CopyObjectRequest.builder()
                .sourceBucket(bucketName)
                .sourceKey(sourceKeyWithEnvironmentPrefix)
                .destinationBucket(bucketName)
                .destinationKey(destinationKeyWithEnvironmentPrefix)
                .metadataDirective(MetadataDirective.COPY)
                .build();

        s3Client.copyObject(copyReq);
    }

    public void copyWithAddedMetadata(String sourceKey, String destinationKey, Map<String, String> addedMetadata) {
        String sourceKeyWithEnvironmentPrefix = ensureEnvironmentPrefixExists(sourceKey);
        String destinationKeyWithEnvironmentPrefix = ensureEnvironmentPrefixExists(destinationKey);

        HeadObjectRequest headReq = HeadObjectRequest.builder()
                .bucket(bucketName)
                .key(sourceKeyWithEnvironmentPrefix)
                .build();
        HeadObjectResponse response = s3Client.headObject(headReq);
        Map<String, String> newMetadata = new HashMap<>(response.metadata());
        newMetadata.putAll(addedMetadata);

        CopyObjectRequest copyReq = CopyObjectRequest.builder()
                .sourceBucket(bucketName)
                .sourceKey(sourceKeyWithEnvironmentPrefix)
                .destinationBucket(bucketName)
                .destinationKey(destinationKeyWithEnvironmentPrefix)
                .metadataDirective(MetadataDirective.REPLACE)
                .metadata(newMetadata)
                .build();

        s3Client.copyObject(copyReq);
    }

    public void delete(String objectKey) {
        String objectKeyWithEnvironmentPrefix = ensureEnvironmentPrefixExists(objectKey);
        DeleteObjectRequest deleteObjectRequest = DeleteObjectRequest.builder()
                .bucket(bucketName)
                .key(objectKeyWithEnvironmentPrefix)
                .build();

        s3Client.deleteObject(deleteObjectRequest);

        log.info("Fact graph {} in bucket {} is deleted", objectKeyWithEnvironmentPrefix, bucketName);
    }

    public boolean doesObjectAlreadyExist(String objectKeyInput) {
        String objectKey = ensureEnvironmentPrefixExists(objectKeyInput);
        try {
            HeadObjectRequest objectRequest = HeadObjectRequest.builder()
                    .key(objectKey)
                    .bucket(bucketName)
                    .build();

            HeadObjectResponse response = s3Client.headObject(objectRequest);
            // null response should only happen in unit test; Exception is thrown when the
            // service cannot find the
            // object.
            return response != null;
        } catch (NoSuchKeyException e) {
            return false;
        }
    }

    public List<DocumentStoreResource> listLocationResources(String baseResourceLocationInput) {
        String baseResourceLocation = ensureEnvironmentPrefixExists(baseResourceLocationInput);
        try {
            ListObjectsRequest listReq = ListObjectsRequest.builder()
                    .bucket(bucketName)
                    .prefix(baseResourceLocation)
                    .build();

            ListObjectsResponse res = s3Client.listObjects(listReq);
            if (res == null) throw new TaxReturnNotFoundResponseStatusException();
            List<S3Object> objects = res.contents();
            List<DocumentStoreResource> docResources = new ArrayList<>();

            // Convert the returned objects into a non-s3 specific return so that it's
            // easier to add an interface and
            // we aren't tied to S3 implementation details
            for (S3Object obj : objects) {
                // Resource ID should be the unique key - location and any file extension
                String resourceId = obj.key().replace(baseResourceLocation, ""); // remove pathing
                resourceId = resourceId.substring(0, resourceId.lastIndexOf(".")); // remove file ext

                DocumentStoreResource doc = new DocumentStoreResource(obj.key(), resourceId, obj.lastModified());
                docResources.add(doc);
            }

            return docResources;
        } catch (S3Exception e) {
            log.error("Error listing S3 location resources", e);
            throw e;
        }
    }

    public InputStream download(String objectKeyInput) throws IOException, DocumentNotFoundException {
        String objectKey = ensureEnvironmentPrefixExists(objectKeyInput);
        GetObjectRequest getObjectRequest =
                GetObjectRequest.builder().bucket(bucketName).key(objectKey).build();
        InputStream objectStream = null;

        try {
            ResponseBytes<GetObjectResponse> objectBytes = s3Client.getObjectAsBytes(getObjectRequest);
            if (objectBytes == null) throw new IOException("Empty file found!");

            objectStream = new ByteArrayInputStream(objectBytes.asByteArray());
        } catch (S3EncryptionClientException s3EncryptionClientException) {
            if (s3EncryptionClientException.getCause().getClass() == NoSuchKeyException.class) {
                throw new DocumentNotFoundException("Could not find document", s3EncryptionClientException);
            }
            throw s3EncryptionClientException;
        }
        return objectStream;
    }

    private String ensureEnvironmentPrefixExists(String objectKey) {
        return StringUtils.prependIfMissing(objectKey, environmentPrefix);
    }
}
