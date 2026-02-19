package gov.irs.directfile.api.config;

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
@ConditionalOnProperty(value = "direct-file.aws.sns.submission-confirmation-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(SnsConfigurationProperties.class)
@AllArgsConstructor
public class SnsClientConfiguration {
    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    public SnsClient snsClient(SnsConfigurationProperties snsConfigurationProperties) {
        return SnsClient.builder()
                .region(Region.of(snsConfigurationProperties.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(snsConfigurationProperties.getEndpoint()))
                .build();
    }
}
