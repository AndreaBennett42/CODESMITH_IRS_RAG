package gov.irs.directfile.models.autoconfigure;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "aws")
public class AWSProperties {
    private String accessKey;
    private String secretKey;
    private String region;
    private String kmsEndpoint;
    private String kmsWrappingKeyArn;
}
