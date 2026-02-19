package gov.irs.directfile.models.autoconfigure;

import java.net.URI;
import java.util.Base64;
import java.util.concurrent.TimeUnit;
import javax.crypto.spec.SecretKeySpec;

import com.amazonaws.encryptionsdk.AwsCrypto;
import com.amazonaws.encryptionsdk.CryptoMaterialsManager;
import com.amazonaws.encryptionsdk.MasterKeyProvider;
import com.amazonaws.encryptionsdk.caching.CachingCryptoMaterialsManager;
import com.amazonaws.encryptionsdk.caching.CryptoMaterialsCache;
import com.amazonaws.encryptionsdk.caching.LocalCryptoMaterialsCache;
import com.amazonaws.encryptionsdk.jce.JceMasterKey;
import com.amazonaws.encryptionsdk.kmssdkv2.AwsKmsMrkAwareMasterKeyProvider;
import com.amazonaws.encryptionsdk.kmssdkv2.RegionalClientSupplier;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.kms.KmsClient;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;

@AutoConfiguration
@Slf4j
@Import(DataEncryptDecrypt.class)
public class EncryptionAutoConfiguration {

    @Configuration
    @EnableConfigurationProperties({AWSProperties.class, AWSCryptoCacheProperties.class})
    @AllArgsConstructor
    public static class AWSConfiguration {
        private final AWSProperties awsProperties;
        private final AWSCryptoCacheProperties awsCryptoCacheProperties;

        @Bean
        @ConditionalOnExpression(
                "'${aws.enabled}' == 'true' and '${aws.default-credentials-provider-chain-enabled}' != 'true'")
        public MasterKeyProvider<?> wrappingKeyProvider() {
            log.info("Database: Setting up KMS for encryption...");
            return AwsKmsMrkAwareMasterKeyProvider.builder()
                    .customRegionalClientSupplier(new KMSClientSupplier())
                    .buildStrict(awsProperties.getKmsWrappingKeyArn());
        }

        @Bean
        @ConditionalOnExpression(
                "'${aws.enabled}' == 'true' and '${aws.default-credentials-provider-chain-enabled}' == 'true'")
        public MasterKeyProvider<?> wrappingKeyProviderDefaultCredentialChain() {
            log.info("Database: Setting up KMS for encryption using default credentials provider chain...");
            return AwsKmsMrkAwareMasterKeyProvider.builder()
                    .customRegionalClientSupplier(new KMSClientSupplierDefaultCredentialsProvider())
                    .buildStrict(awsProperties.getKmsWrappingKeyArn());
        }

        @Bean
        public AwsCrypto awsCrypto() {
            return AwsCrypto.standard();
        }

        @Bean
        public CryptoMaterialsCache awsCryptoMaterialsCache() {
            return new LocalCryptoMaterialsCache(awsCryptoCacheProperties.getMaxItems());
        }

        @Bean
        public CryptoMaterialsManager awsCryptoMaterialsManager(
                MasterKeyProvider<?> wrappingKeyProvider, CryptoMaterialsCache cryptoMaterialsCache) {
            return CachingCryptoMaterialsManager.newBuilder()
                    .withMasterKeyProvider(wrappingKeyProvider)
                    .withCache(cryptoMaterialsCache)
                    .withMaxAge(awsCryptoCacheProperties.getMaxAgeSeconds(), TimeUnit.SECONDS)
                    .withMessageUseLimit(awsCryptoCacheProperties.getMessageUseLimit())
                    .build();
        }

        private class KMSClientSupplier implements RegionalClientSupplier {

            @Override
            public KmsClient getClient(Region region) {
                return KmsClient.builder()
                        .region(Region.of(awsProperties.getRegion()))
                        .credentialsProvider(StaticCredentialsProvider.create(
                                AwsBasicCredentials.create(awsProperties.getAccessKey(), awsProperties.getSecretKey())))
                        .endpointOverride(URI.create(awsProperties.getKmsEndpoint()))
                        .build();
            }
        }

        private class KMSClientSupplierDefaultCredentialsProvider implements RegionalClientSupplier {

            @Override
            public KmsClient getClient(Region region) {
                return KmsClient.builder()
                        .region(Region.of(awsProperties.getRegion()))
                        .endpointOverride(URI.create(awsProperties.getKmsEndpoint()))
                        .build();
            }
        }
    }

    /**
     * Local Encryption: This configuration is meant to be used for local development and testing only.
     * In deployed environments, we use AWS KMS for encryption, which requires separate configuration.
     */
    @Configuration
    @EnableConfigurationProperties(LocalEncryptionProperties.class)
    public static class LocalEncryptionConfiguration {
        @Autowired
        LocalEncryptionProperties localEncryptionProperties;

        @Bean
        @ConditionalOnProperty(prefix = "aws", name = "enabled", havingValue = "false")
        public MasterKeyProvider<?> wrappingKeyProvider() {
            log.warn("Database: Using local encryption without AWS KMS. Not appropriate for deployed environments!");
            try {
                String key = checkForLocalWrappingKey();
                SecretKeySpec secretKeySpec =
                        new SecretKeySpec(Base64.getDecoder().decode(key), "AES");
                return JceMasterKey.getInstance(secretKeySpec, "local", "local", "AES/GCM/NoPadding");
            } catch (IllegalArgumentException e) {
                String message =
                        """
                        LOCAL WRAPPING KEY NOT SET: \
                        Can't run the application because the local wrapping key isn't set or isn't a valid base64 String. \
                        See ONBOARDING.md for more instructions on setting up your local environment variables.
                        """;
                log.error(message);
                throw new RuntimeException(message, e);
            }
        }

        private String checkForLocalWrappingKey() {
            String key = localEncryptionProperties.getLocalWrappingKey();
            if (StringUtils.isBlank(key)) {
                throw new IllegalArgumentException("local wrapping key is blank");
            }

            return key;
        }
    }
}
