package gov.irs.directfile.api.config;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;

@Configuration
@EnableConfigurationProperties(S3ConfigurationProperties.class)
public class AwsCredentialsConfiguration {

    @Bean
    @ConditionalOnProperty(
            name = "direct-file.aws.default-credentials-provider-chain-enabled",
            havingValue = "false",
            matchIfMissing = true)
    public AwsCredentialsProvider staticCredentialsProvider(S3ConfigurationProperties s3ConfigurationProperties) {
        return StaticCredentialsProvider.create(AwsBasicCredentials.create(
                s3ConfigurationProperties.getCredentials().getAccessKey(),
                s3ConfigurationProperties.getCredentials().getSecretKey()));
    }

    @Bean
    @ConditionalOnProperty(name = "direct-file.aws.default-credentials-provider-chain-enabled", havingValue = "true")
    public AwsCredentialsProvider defaultAWSCredentialsProvider() {
        return DefaultCredentialsProvider.create();
    }
}
