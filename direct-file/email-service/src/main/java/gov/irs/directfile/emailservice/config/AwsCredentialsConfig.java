package gov.irs.directfile.emailservice.config;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;

@Configuration
@EnableConfigurationProperties(EmailServiceConfigurationProperties.class)
public class AwsCredentialsConfig {

    @Bean
    @ConditionalOnProperty(
            name = "email-service.messageQueue.default-credentials-provider-chain-enabled",
            havingValue = "false",
            matchIfMissing = true)
    public AwsCredentialsProvider staticAWSCredentialsProvider(
            EmailServiceConfigurationProperties emailServiceConfigurationProperties) {
        return StaticCredentialsProvider.create(AwsBasicCredentials.create(
                emailServiceConfigurationProperties.getMessageQueue().getAccessKey(),
                emailServiceConfigurationProperties.getMessageQueue().getSecretKey()));
    }

    @Bean
    @ConditionalOnProperty(
            name = "email-service.messageQueue.default-credentials-provider-chain-enabled",
            havingValue = "true")
    public AwsCredentialsProvider defaultAWSCredentialsProvider() {
        return DefaultCredentialsProvider.create();
    }
}
