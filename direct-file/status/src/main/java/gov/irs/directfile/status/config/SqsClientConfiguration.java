package gov.irs.directfile.status.config;

import java.net.URI;

import lombok.AllArgsConstructor;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sqs.SqsClient;

@Configuration
@EnableConfigurationProperties(MessageQueueConfiguration.class)
@AllArgsConstructor
public class SqsClientConfiguration {
    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    public SqsClient sqsClient(MessageQueueConfiguration messageQueueConfiguration) {
        return SqsClient.builder()
                .region(Region.of(messageQueueConfiguration.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(messageQueueConfiguration.getEndpoint()))
                .build();
    }
}
