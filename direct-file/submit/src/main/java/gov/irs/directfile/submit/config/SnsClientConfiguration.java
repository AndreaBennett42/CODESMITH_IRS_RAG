package gov.irs.directfile.submit.config;

import java.net.URI;

import lombok.AllArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sns.SnsClient;

@Configuration
@ConditionalOnProperty(value = "submit.sns.submission-confirmation-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(SnsConfig.class)
@AllArgsConstructor
public class SnsClientConfiguration {

    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    public SnsClient snsClient(SnsConfig snsConfig) {
        return SnsClient.builder()
                .region(Region.of(snsConfig.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(snsConfig.getEndpoint()))
                .build();
    }
}
