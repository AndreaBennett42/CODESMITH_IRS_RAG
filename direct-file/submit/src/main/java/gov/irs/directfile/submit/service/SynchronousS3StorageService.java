package gov.irs.directfile.submit.service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.encryption.s3.S3EncryptionClient;

import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.domain.DocumentStoreResource;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@Service
@Slf4j
@SuppressFBWarnings(
        value = {"DM_DEFAULT_ENCODING", "NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@SuppressWarnings({"PMD.AvoidReassigningParameters", "PMD.MissingOverride"})
public class SynchronousS3StorageService implements ISynchronousDocumentStoreService {
    private final S3EncryptionClient s3Client;

    private final String bucketName;

    private final String environmentPrefix;

    @Autowired
    public SynchronousS3StorageService(
            S3EncryptionClient s3Client, @Value("${submit.documentstore.bucket}") String bucketName, Config config) {
        this.s3Client = s3Client;
        this.bucketName = bucketName;
        this.environmentPrefix = config.getDocumentStore().getEnvironmentPrefix();
    }

    @Override
    public String write(String objectKey, InputStream payloadStream) throws IOException {
        objectKey = ensureEnvironmentPrefixExists(objectKey);

        PutObjectRequest putObjectRequest =
                PutObjectRequest.builder().bucket(bucketName).key(objectKey).build();
        RequestBody requestBody = RequestBody.fromBytes(IOUtils.toByteArray(payloadStream));
        PutObjectResponse putObjectResponse = s3Client.putObject(putObjectRequest, requestBody);
        return putObjectResponse.eTag();
    }

    @Override
    public String write(String objectKey, String content) {
        objectKey = ensureEnvironmentPrefixExists(objectKey);

        PutObjectRequest putObjectRequest =
                PutObjectRequest.builder().bucket(bucketName).key(objectKey).build();
        RequestBody requestBody = RequestBody.fromString(content, StandardCharsets.UTF_8);
        PutObjectResponse putObjectResponse = s3Client.putObject(putObjectRequest, requestBody);
        return putObjectResponse.eTag();
    }

    @Override
    public List<DocumentStoreResource> getObjectKeys(String prefix) {
        prefix = ensureEnvironmentPrefixExists(prefix);
        String emptyContinuationToken = null;
        return getObjectKeysRecursive(prefix, new ArrayList<>(), emptyContinuationToken);
    }

    @Override
    public void deleteObjects(List<String> keys) {
        List<ObjectIdentifier> identifiers = keys.stream()
                .map(key -> ObjectIdentifier.builder().key(key).build())
                .toList();
        DeleteObjectsRequest deleteObjectsRequest = DeleteObjectsRequest.builder()
                .bucket(bucketName)
                .delete(Delete.builder().objects(identifiers).build())
                .build();
        s3Client.deleteObjects(deleteObjectsRequest);
    }

    private List<DocumentStoreResource> getObjectKeysRecursive(
            String prefix, List<DocumentStoreResource> results, String continuationToken) {
        ListObjectsV2Request listObjectsV2Request = ListObjectsV2Request.builder()
                .bucket(bucketName)
                .prefix(prefix)
                .continuationToken(continuationToken)
                .build();
        ListObjectsV2Response response = s3Client.listObjectsV2(listObjectsV2Request);

        List<DocumentStoreResource> page = response.contents().stream()
                .map(s3Object -> new DocumentStoreResource(s3Object.key(), s3Object.eTag(), s3Object.lastModified()))
                .toList();
        results.addAll(page);

        if (response.isTruncated()) {
            return getObjectKeysRecursive(prefix, results, response.nextContinuationToken());
        } else {
            return results;
        }
    }

    public Optional<String> getMostRecentFolderForPrefix(String prefix) {
        prefix = ensureEnvironmentPrefixExists(prefix);

        ListObjectsV2Request listObjectsV2Request = ListObjectsV2Request.builder()
                .bucket(bucketName)
                .prefix(prefix)
                .delimiter("/")
                .build();
        ListObjectsV2Response response = s3Client.listObjectsV2(listObjectsV2Request);
        return response.commonPrefixes().stream()
                .map(CommonPrefix::prefix)
                .sorted()
                .max(Comparator.comparing(Function.identity()));
    }

    public List<String> getSubFolders(String objectKey) {
        objectKey = ensureEnvironmentPrefixExists(objectKey);

        return getSubFoldersRecursive(objectKey, new ArrayList<>(), null);
    }

    private List<String> getSubFoldersRecursive(String objectKey, List<String> results, String continuationToken) {
        objectKey = ensureEnvironmentPrefixExists(objectKey);
        ListObjectsV2Request listObjectsV2Request = ListObjectsV2Request.builder()
                .bucket(bucketName)
                .prefix(objectKey)
                .continuationToken(continuationToken)
                .delimiter("/")
                .build();

        ListObjectsV2Response response = s3Client.listObjectsV2(listObjectsV2Request);

        List<String> pageResults =
                response.commonPrefixes().stream().map(CommonPrefix::prefix).toList();
        results.addAll(pageResults);

        if (response.isTruncated()) {
            return getSubFoldersRecursive(objectKey, results, response.nextContinuationToken());
        } else {
            return results;
        }
    }

    public Optional<DocumentStoreResource> getLeastRecentModifiedResourceForPrefix(String objectKey) {
        objectKey = ensureEnvironmentPrefixExists(objectKey);
        ListObjectsV2Request listObjectsV2Request = ListObjectsV2Request.builder()
                .bucket(bucketName)
                .prefix(objectKey)
                .delimiter("/")
                .build();

        ListObjectsV2Response response = s3Client.listObjectsV2(listObjectsV2Request);

        return response.contents().stream()
                .map(s3Object -> new DocumentStoreResource(s3Object.key(), s3Object.eTag(), s3Object.lastModified()))
                .min(Comparator.comparing(DocumentStoreResource::getLastModified));
    }

    @Override
    public String getObjectAsString(String objectKey) throws IOException {
        objectKey = ensureEnvironmentPrefixExists(objectKey);
        GetObjectRequest getObjectRequest =
                GetObjectRequest.builder().bucket(bucketName).key(objectKey).build();

        ResponseBytes<GetObjectResponse> getObjectResponse = s3Client.getObjectAsBytes(getObjectRequest);
        return new String(getObjectResponse.asByteArray());
    }

    private String ensureEnvironmentPrefixExists(String objectKey) {
        return StringUtils.prependIfMissing(objectKey, environmentPrefix);
    }

    @Override
    public void copyObject(DocumentStoreResource documentStoreResource, String destinationKey) {
        destinationKey = ensureEnvironmentPrefixExists(destinationKey);
        CopyObjectRequest copyObjectRequest = CopyObjectRequest.builder()
                .sourceBucket(bucketName)
                .sourceKey(documentStoreResource.getFullLocation())
                .destinationKey(destinationKey)
                .destinationBucket(bucketName)
                .build();

        s3Client.copyObject(copyObjectRequest);
    }

    @Override
    public void Setup(Config config) throws Throwable {}
}
