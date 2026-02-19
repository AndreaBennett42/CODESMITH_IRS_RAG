package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;

@AllArgsConstructor
@Getter
@ConfigurationProperties
public class EncryptionConfig {
    @Value("${aws.kmsEndpoint:#{null}}")
    private String kmsEndpoint;

    @Value("${aws.region:#{null}}")
    private String region;

    @Value("${direct-file.local-encryption.local-wrapping-key:#{null}}")
    private String localWrappingKey;
}
