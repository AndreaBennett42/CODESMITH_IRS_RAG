package gov.irs.directfile.status.config;

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
@ConditionalOnProperty(value = "status.sns.status-change-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(SnsConfiguration.class)
@AllArgsConstructor
public class SnsClientConfiguration {
    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    public SnsClient snsClient(SnsConfiguration snsConfiguration) {
        return SnsClient.builder()
                .region(Region.of(snsConfiguration.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(snsConfiguration.getEndpoint()))
                .build();
    }
}
