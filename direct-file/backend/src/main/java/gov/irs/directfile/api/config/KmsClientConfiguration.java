package gov.irs.directfile.api.config;

import java.net.URI;

import lombok.AllArgsConstructor;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.kms.KmsClient;

@Configuration
@EnableConfigurationProperties({AwsConfigurationProperties.class, S3ConfigurationProperties.class})
@AllArgsConstructor
public class KmsClientConfiguration {
    private final AwsConfigurationProperties awsConfigurationProperties;
    private final S3ConfigurationProperties s3ConfigurationProperties;
    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    @Profile("aws")
    public KmsClient regionalKmsClient() {
        return KmsClient.builder()
                .region(Region.of(s3ConfigurationProperties.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(awsConfigurationProperties.getKmsEndpoint()))
                .build();
    }
}
