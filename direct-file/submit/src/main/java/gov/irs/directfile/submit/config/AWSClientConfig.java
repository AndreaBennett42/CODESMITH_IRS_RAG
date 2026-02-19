package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "aws")
@AllArgsConstructor
@Getter
public class AWSClientConfig {
    private String accessKey;
    private String secretKey;
}
