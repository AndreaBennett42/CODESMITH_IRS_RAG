package gov.irs.directfile.models.autoconfigure;

import java.security.SecureRandom;
import java.util.Base64;

import com.amazonaws.encryptionsdk.MasterKeyProvider;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.springframework.beans.BeanInstantiationException;
import org.springframework.beans.factory.NoSuchBeanDefinitionException;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;

import static org.assertj.core.api.Assertions.assertThat;

public class EncryptionAutoConfigurationTest {
    private final ApplicationContextRunner contextRunner =
            new ApplicationContextRunner().withConfiguration(AutoConfigurations.of(EncryptionAutoConfiguration.class));

    @Test
    @Disabled
    void awsPropertiesCanBeConfigured() {
        this.contextRunner
                .withPropertyValues(
                        "aws.enabled=true",
                        "aws.default-credentials-provider-chain-enabled=false",
                        "aws.kmsWrappingKeyArn=test")
                .run((context) -> {
                    // This only fails because AWS's strict configuration will fail with invalid ARNs
                    // this shows that both of the configured properties are detected and that the failure is
                    // specifically due to the ARN.
                    assertThat(context.getStartupFailure().getMessage()).contains("ARN");
                });
    }

    @Test
    void localEncryptionPropertiesCanBeConfigured() {
        this.contextRunner
                .withPropertyValues(
                        "aws.enabled=false",
                        "direct-file.local-encryption.localWrappingKey=" + generateLocalKey(),
                        "aws.crypto-cache.message-use-limit=10",
                        "aws.crypto-cache.max-age-seconds=10",
                        "aws.crypto-cache.max-items=10")
                .run((context) -> {
                    assertThat(context.getBean(MasterKeyProvider.class)).isNotNull();
                });
    }

    @Test
    void missingAwsConfigurationCausesException() {
        this.contextRunner.run((context) -> {
            assertThat(context).hasFailed();
            assertThat(context.getStartupFailure().getCause()).isInstanceOf(NoSuchBeanDefinitionException.class);
        });
    }

    @Test
    void invalidConfigurationCausesException() {
        this.contextRunner
                .withPropertyValues(
                        "aws.enabled=1", "direct-file.local-encryption.localWrappingKey=" + generateLocalKey())
                .run((context) -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure().getCause())
                            .isInstanceOf(NoSuchBeanDefinitionException.class);
                });
    }

    @Test
    void givenAwsDisabledConfiguration_whenMissingWrappingKeyConfiguration_thenCausesException() {
        this.contextRunner.withPropertyValues("aws.enabled=false").run((context) -> {
            assertThat(context).hasFailed();
            assertThat(context.getStartupFailure().getCause().getMessage()).contains("LOCAL WRAPPING KEY NOT SET");
        });
    }

    @Test
    void givenAwsEnabledConfiguration_whenMissingWrappingKeyConfiguration_thenCausesException() {
        this.contextRunner
                .withPropertyValues("aws.enabled=true", "aws.default-credentials-provider-chain-enabled=false")
                .run((context) -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure().getCause()).isInstanceOf(BeanInstantiationException.class);
                });
    }

    @Test
    void givenAwsDefaultCredentialConfiguration_whenMissingWrappingKeyConfiguration_thenCausesException() {
        this.contextRunner
                .withPropertyValues("aws.enabled=true", "aws.default-credentials-provider-chain-enabled=true")
                .run((context) -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure().getCause()).isInstanceOf(BeanInstantiationException.class);
                });
    }

    @Test
    void givenValidConfiguration_thenEncryptionBeanLoads() {
        this.contextRunner
                .withPropertyValues(
                        "aws.enabled=false",
                        "direct-file.local-encryption.localWrappingKey=" + generateLocalKey(),
                        "aws.crypto-cache.message-use-limit=10",
                        "aws.crypto-cache.max-age-seconds=10",
                        "aws.crypto-cache.max-items=10")
                .run((context) -> {
                    assertThat(context.getBean(DataEncryptDecrypt.class)).isNotNull();
                });
    }

    public String generateLocalKey() {
        byte[] secureRandomKeyBytes = new byte[256 / 8];
        SecureRandom secureRandom = new SecureRandom();
        secureRandom.nextBytes(secureRandomKeyBytes);
        return Base64.getEncoder().encodeToString(secureRandomKeyBytes);
    }
}
