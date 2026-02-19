package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;

@Configuration
@AllArgsConstructor
@EnableConfigurationProperties(AWSClientConfig.class)
public class AWSCredentialsConfig {

    private final AWSClientConfig awsClientConfig;

    @Bean
    @ConditionalOnProperty(
            name = "aws.default-credentials-provider-chain-enabled",
            havingValue = "false",
            matchIfMissing = true)
    public AwsCredentialsProvider staticAWSCredentialsProvider() {
        return StaticCredentialsProvider.create(
                AwsBasicCredentials.create(awsClientConfig.getAccessKey(), awsClientConfig.getSecretKey()));
    }

    @Bean
    @ConditionalOnProperty(name = "aws.default-credentials-provider-chain-enabled", havingValue = "true")
    public AwsCredentialsProvider defaultAWSCredentialsProvider() {
        return DefaultCredentialsProvider.create();
    }
}
