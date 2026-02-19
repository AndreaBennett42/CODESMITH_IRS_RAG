package gov.irs.directfile.api.dataimport.gating;

import java.io.IOException;
import java.nio.charset.Charset;

import com.fasterxml.jackson.databind.DeserializationFeature;
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
import gov.irs.directfile.api.config.DataImportGatingConfigurationProperties;
import gov.irs.directfile.api.config.DataImportGatingS3Properties;
import gov.irs.directfile.api.config.RedisConfiguration;
import gov.irs.directfile.api.dataimport.exception.DataImportException;

@Slf4j
@Service
@EnableConfigurationProperties({DataImportGatingS3Properties.class, DataImportGatingConfigurationProperties.class})
public class DataImportGatingConfigService {

    private final S3Client s3Client;
    private final DataImportGatingS3Properties gatingS3Config;
    private final ObjectMapper deserializationObjectMapper;
    private final CacheService cacheService;

    public DataImportGatingConfigService(
            @Qualifier("s3WithoutEncryption") S3Client s3Client,
            DataImportGatingS3Properties dataImportGatingConfig,
            CacheService cacheService) {
        this.s3Client = s3Client;
        this.gatingS3Config = dataImportGatingConfig;
        this.cacheService = cacheService;

        // Deserialize from kebab case as the properties appear in the data import behavior file
        // Serialization is handled by our default object mapper in the controller (camel case)
        ObjectMapper deserializationObjectMapper = new ObjectMapper();
        deserializationObjectMapper.setPropertyNamingStrategy(new PropertyNamingStrategies.KebabCaseStrategy());
        deserializationObjectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        deserializationObjectMapper.findAndRegisterModules();
        this.deserializationObjectMapper = deserializationObjectMapper;
    }

    public DataImportGatingConfig getGatingS3Config() {
        try {
            return getDataImportGatingConfigWithCache(gatingS3Config.getDataImportGatingObject());
        } catch (Exception e) {
            log.error(
                    "Error occurred to retrieve data-import-gating config file {} in bucket {}. Exception: {}. Error: {}",
                    gatingS3Config.getDataImportGatingObject(),
                    gatingS3Config.getDataImportGatingBucket(),
                    e.getClass().getName(),
                    e.getMessage());
            return null;
        }
    }

    /*
     * Cache/retrieve objects related to data-import-gating, e.g. the email allowlist
     */
    public String getDataImportGatingObjectAsString(String objectKey) {
        try {
            return getDataImportGatingObjectAsStringWithCache(objectKey);
        } catch (Exception e) {
            log.error(
                    "Error during data-import-gating object {} retrieval. Exception: {}. Error: {}",
                    objectKey,
                    e.getClass().getName(),
                    e.getMessage());
            throw new DataImportException("Error retrieving data-import-gating object " + objectKey, e);
        }
    }

    private String getDataImportGatingObjectAsStringWithCache(String objectKey) {
        String dataImportGatingObject =
                cacheService.get(RedisConfiguration.DATA_IMPORT_GATING_CACHE_NAME, objectKey, String.class);
        if (dataImportGatingObject != null) {
            return dataImportGatingObject;
        }

        dataImportGatingObject = getObjectAsString(objectKey);
        cacheService.set(
                RedisConfiguration.DATA_IMPORT_GATING_CACHE_NAME,
                objectKey,
                dataImportGatingObject,
                gatingS3Config.getDataImportGatingExpiration());
        return dataImportGatingObject;
    }

    private String getObjectAsString(String objectKey) {
        return getObject(objectKey).asUtf8String();
    }

    private byte[] getObjectBytes(String objectKey) {
        return getObject(objectKey).asByteArray();
    }

    private ResponseBytes<GetObjectResponse> getObject(String objectKey) {
        String objectKeyWithEnv = ensureEnvironmentPrefixExists(objectKey);
        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(gatingS3Config.getDataImportGatingBucket())
                .key(objectKeyWithEnv)
                .build();
        ResponseBytes<GetObjectResponse> getObjectResponse = s3Client.getObjectAsBytes(getObjectRequest);
        log.info("Successfully retrieved {} from S3", objectKeyWithEnv);
        return getObjectResponse;
    }

    private DataImportGatingConfig getDataImportGatingConfigWithCache(String objKey) throws IOException {
        DataImportGatingConfig config = cacheService.get(
                RedisConfiguration.DATA_IMPORT_GATING_CACHE_NAME, objKey, DataImportGatingConfig.class);
        if (config != null) {
            return config;
        }

        byte[] configBytes = getObjectBytes(objKey);
        if (configBytes == null) {
            return null;
        }

        log.info("Data Import Gating Config: {}", new String(configBytes, Charset.forName("UTF-8")));

        config = deserializationObjectMapper.readValue(configBytes, DataImportGatingConfig.class);

        cacheService.set(
                RedisConfiguration.DATA_IMPORT_GATING_CACHE_NAME,
                objKey,
                config,
                gatingS3Config.getDataImportGatingExpiration());
        return config;
    }

    private String ensureEnvironmentPrefixExists(String objectKey) {
        return StringUtils.prependIfMissing(objectKey, gatingS3Config.getEnvironmentPrefix());
    }
}
