package gov.irs.directfile.stateapi.configuration;

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
            name = "aws.s3.default-credentials-provider-chain-enabled",
            havingValue = "false",
            matchIfMissing = true)
    public AwsCredentialsProvider staticCredentialsProvider(S3ConfigurationProperties s3ConfigurationProperties) {
        return StaticCredentialsProvider.create(AwsBasicCredentials.create(
                s3ConfigurationProperties.getAccessKey(), s3ConfigurationProperties.getSecretKey()));
    }

    @Bean
    @ConditionalOnProperty(name = "aws.s3.default-credentials-provider-chain-enabled", havingValue = "true")
    public AwsCredentialsProvider defaultAWSCredentialsProvider() {
        return DefaultCredentialsProvider.create();
    }
}
