package gov.irs.directfile.api.featureflags;

import java.io.IOException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;

import gov.irs.directfile.api.cache.CacheService;
import gov.irs.directfile.api.config.RedisConfiguration;
import gov.irs.directfile.api.errors.FeatureFlagException;

@Slf4j
@Service
@EnableConfigurationProperties({FeatureFlagConfigurationProperties.class})
public class FeatureFlagService {
    private final S3Client s3Client;
    private final FeatureFlagConfigurationProperties featureFlagConfigurationProperties;
    private final ObjectMapper deserializationObjectMapper;
    private final CacheService cacheService;

    public FeatureFlagService(
            @Qualifier("s3WithoutEncryption") S3Client s3Client,
            FeatureFlagConfigurationProperties featureFlagConfigurationProperties,
            CacheService cacheService) {
        this.s3Client = s3Client;
        this.featureFlagConfigurationProperties = featureFlagConfigurationProperties;
        this.cacheService = cacheService;

        // Deserialize from kebab case as the properties appear in the feature flags file
        // Serialization is handled by our default object mapper in the controller (camel case)
        ObjectMapper deserializationObjectMapper = new ObjectMapper();
        deserializationObjectMapper.setPropertyNamingStrategy(new PropertyNamingStrategies.KebabCaseStrategy());
        this.deserializationObjectMapper = deserializationObjectMapper;
    }

    private byte[] getObjectBytes(String objKey) {
        String objectKey = ensureEnvironmentPrefixExists(objKey);
        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(featureFlagConfigurationProperties.getFeatureFlagsBucket())
                .key(objectKey)
                .build();
        ResponseBytes<GetObjectResponse> getObjectResponse = s3Client.getObjectAsBytes(getObjectRequest);
        return getObjectResponse.asByteArray();
    }

    private FeatureFlags getFeatureFlagsWithCache(String objKey) throws IOException {
        FeatureFlags flags = cacheService.get(RedisConfiguration.FEATURE_FLAG_CACHE_NAME, objKey, FeatureFlags.class);
        if (flags != null) {
            return flags;
        }

        byte[] featureConfigBytes = getObjectBytes(featureFlagConfigurationProperties.getFeatureFlagsObject());
        flags = deserializationObjectMapper.readValue(featureConfigBytes, FeatureFlags.class);
        cacheService.set(
                RedisConfiguration.FEATURE_FLAG_CACHE_NAME,
                objKey,
                flags,
                featureFlagConfigurationProperties.getFeatureFlagsExpiration());
        return flags;
    }

    private String ensureEnvironmentPrefixExists(String objectKey) {
        return StringUtils.prependIfMissing(objectKey, featureFlagConfigurationProperties.getEnvironmentPrefix());
    }

    public FeatureFlags getFeatureFlags() {
        try {
            return getFeatureFlagsWithCache(featureFlagConfigurationProperties.getFeatureFlagsObject());
        } catch (Exception e) {
            log.error("Error during feature flag configuration retrieval: {}", e.getMessage());
            throw new FeatureFlagException("Error retrieving feature flags", e);
        }
    }

    /*
     * Cache/retrieve objects related to features, e.g. the email allowlist
     */
    public String getFeatureObjectAsString(String objectKey) {
        try {
            return getFeatureObjectAsStringWithCache(objectKey);
        } catch (Exception e) {
            log.error("Error during feature object retrieval: {}", e.getMessage());
            throw new FeatureFlagException("Error retrieving feature object", e);
        }
    }

    private String getFeatureObjectAsStringWithCache(String objectKey) {
        String featureObject = cacheService.get(RedisConfiguration.FEATURE_FLAG_CACHE_NAME, objectKey, String.class);
        if (featureObject != null) {
            return featureObject;
        }

        featureObject = getObjectAsString(objectKey);
        cacheService.set(
                RedisConfiguration.FEATURE_FLAG_CACHE_NAME,
                objectKey,
                featureObject,
                featureFlagConfigurationProperties.getFeatureFlagsExpiration());
        return featureObject;
    }

    private String getObjectAsString(String objectKey) {
        String objectKeyWithEnv = ensureEnvironmentPrefixExists(objectKey);
        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(featureFlagConfigurationProperties.getFeatureFlagsBucket())
                .key(objectKeyWithEnv)
                .build();
        ResponseBytes<GetObjectResponse> getObjectResponse = s3Client.getObjectAsBytes(getObjectRequest);
        return getObjectResponse.asUtf8String();
    }
}
