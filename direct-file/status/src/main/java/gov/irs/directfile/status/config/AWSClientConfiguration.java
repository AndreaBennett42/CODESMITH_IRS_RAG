package gov.irs.directfile.status.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "aws")
@AllArgsConstructor
@Getter
public class AWSClientConfiguration {
    private String accessKey;
    private String secretKey;
}
