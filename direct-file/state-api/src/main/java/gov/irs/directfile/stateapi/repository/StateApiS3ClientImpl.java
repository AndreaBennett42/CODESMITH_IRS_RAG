package gov.irs.directfile.stateapi.repository;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Pattern;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.core.async.AsyncResponseTransformer;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.configuration.S3ConfigurationProperties;
import gov.irs.directfile.stateapi.configuration.XmlSanitizedConfigurationProperties;
import gov.irs.directfile.stateapi.exception.StateApiException;

@Component
@Slf4j
@SuppressWarnings({"PMD.AvoidReassigningLoopVariables"})
public class StateApiS3ClientImpl implements StateApiS3Client {
    private final S3AsyncClient s3Client;
    private final S3AsyncEncryptionClient s3EncryptionClient;

    private final S3ConfigurationProperties s3ConfigurationProperties;

    private final XmlSanitizedConfigurationProperties xmlSanitizedConfigurationProperties;

    private List<Pattern> excludedPatterns = new ArrayList<>();

    @Autowired
    public StateApiS3ClientImpl(
            S3AsyncClient s3Client,
            S3AsyncEncryptionClient s3EncryptionClient,
            S3ConfigurationProperties s3ConfigurationProperties,
            XmlSanitizedConfigurationProperties xmlSanitizedConfigurationProperties) {
        this.s3Client = s3Client;
        this.s3EncryptionClient = s3EncryptionClient;
        this.s3ConfigurationProperties = s3ConfigurationProperties;
        this.xmlSanitizedConfigurationProperties = xmlSanitizedConfigurationProperties;

        setExcludedPatterns();
    }

    private void setExcludedPatterns() {
        log.info("xml headers allowed: {}", xmlSanitizedConfigurationProperties.getAllowedHeaders());
        var excludedTags = xmlSanitizedConfigurationProperties.getExcludedTags();
        log.info("xml tags excluded: {}", excludedTags);
        if (excludedTags != null) {
            for (String header : excludedTags) {
                String formattedHeader = header.trim();
                String regex = "<" + formattedHeader + ">[\\s\\S]*?</" + formattedHeader + ">";
                excludedPatterns.add(Pattern.compile(regex));
            }
        }

        // remove all blank lines caused by xml node manipulation
        excludedPatterns.add(Pattern.compile("(?m)^[ \\t]*\\r?\\n"));
    }

    @Override
    public Mono<InputStream> getCert(String certName) {
        log.info("enter getCert()...for cert {}", certName);

        String s3Prefix = s3ConfigurationProperties.getPrefix();
        String objectKey = (StringUtils.isNotBlank(s3Prefix) ? s3Prefix + "/" : "") + certName;

        log.info("checking for cert at path {}", objectKey);

        return getUnencryptedS3Object(
                        s3ConfigurationProperties.getCertBucketName(),
                        objectKey,
                        StateApiErrorCode.E_CERTIFICATE_NOT_FOUND)
                .map(s -> new ByteArrayInputStream(s.getBytes(StandardCharsets.UTF_8)));
    }

    @Override
    public Mono<String> getTaxReturnXml(int taxYear, UUID taxReturnId, String submissionId) {
        String objectKey = generateSubmissionLocationObjectKey(taxYear, taxReturnId, submissionId);

        return getEncryptedS3Object(
                        s3ConfigurationProperties.getTaxReturnXmlBucketName(),
                        objectKey,
                        StateApiErrorCode.E_TAX_RETURN_NOT_FOUND)
                .flatMap(xmlString -> {
                    log.info("getSanitizedXml for: fillingYear={}, taxReturnId={}", taxYear, taxReturnId);

                    try {
                        String sanitizedXml = getXmlString(xmlString);
                        return Mono.just(sanitizedXml);
                    } catch (Exception e) {
                        log.error("getSanitizedXml failed, {}, {}", e.getClass().getName(), e.getMessage(), e);
                        return Mono.error(new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR));
                    }
                });
    }

    private String getXmlString(String ignoredXmlString) {
        return "<xml>";
    }

    private String generateSubmissionLocationObjectKey(int taxFilingYear, UUID taxReturnId, String submissionId) {
        String objectPath = taxFilingYear + "/taxreturns/" + taxReturnId + "/submissions/" + submissionId + ".xml";

        String s3Prefix = s3ConfigurationProperties.getPrefix();
        if (StringUtils.isNotBlank(s3Prefix)) {
            return s3Prefix + "/" + objectPath;
        }

        return objectPath;
    }

    private Mono<String> getEncryptedS3Object(String theBucketName, String objectKey, StateApiErrorCode ec) {
        return getS3Object(theBucketName, objectKey, ec, s3EncryptionClient);
    }

    private Mono<String> getUnencryptedS3Object(String theBucketName, String objectKey, StateApiErrorCode ec) {
        return getS3Object(theBucketName, objectKey, ec, s3Client);
    }

    private Mono<String> getS3Object(
            String theBucketName, String objectKey, StateApiErrorCode ec, S3AsyncClient s3Client) {
        log.info("enter getS3Object()...for bucket: {}, key: {}", theBucketName, objectKey);

        GetObjectRequest objectRequest =
                GetObjectRequest.builder().bucket(theBucketName).key(objectKey).build();

        CompletableFuture<ResponseBytes<GetObjectResponse>> futureGet =
                s3Client.getObject(objectRequest, AsyncResponseTransformer.toBytes());

        return Mono.fromFuture(() -> futureGet)
                .switchIfEmpty(Mono.defer(() -> {
                    log.error(
                            "getS3Object() failed, cannot find s3 object, bucket: {}, key: {}",
                            theBucketName,
                            objectKey);
                    return Mono.error(new StateApiException(ec));
                }))
                .onErrorMap(NoSuchKeyException.class, e -> {
                    log.error(
                            "getS3Object() failed, cannot find s3 object, bucket: {}, key: {}",
                            theBucketName,
                            objectKey);
                    return new StateApiException(ec);
                })
                .onErrorMap(e -> !(e instanceof StateApiException), e -> {
                    log.error(
                            "getS3Object() failed, bucket: {}, key: {}, {}, error: {}",
                            theBucketName,
                            objectKey,
                            e.getClass().getName(),
                            e.getMessage());
                    return new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                })
                .map(responseBytes -> {
                    ByteBuffer byteBuffer = responseBytes.asByteBuffer();
                    byte[] byteArray = new byte[byteBuffer.remaining()];
                    byteBuffer.get(byteArray);
                    log.info(
                            "object retrieved from S3, length: {}, bucket: {}, key: {}",
                            byteArray.length,
                            theBucketName,
                            objectKey);
                    return new String(byteArray, StandardCharsets.UTF_8);
                });
    }
}
