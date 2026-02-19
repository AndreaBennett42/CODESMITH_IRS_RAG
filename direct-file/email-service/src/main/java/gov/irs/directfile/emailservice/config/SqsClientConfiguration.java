package gov.irs.directfile.emailservice.config;

import java.net.URI;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sqs.SqsClient;

// TOOD refactor into custom starter
@Configuration
@EnableConfigurationProperties(EmailServiceConfigurationProperties.class)
public class SqsClientConfiguration {

    @Bean
    public SqsClient sqsClient(
            EmailServiceConfigurationProperties configProps, AwsCredentialsProvider awsCredentialsProvider) {
        return SqsClient.builder()
                .region(Region.of(configProps.getMessageQueue().getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(configProps.getMessageQueue().getEndpoint()))
                .build();
    }
}
