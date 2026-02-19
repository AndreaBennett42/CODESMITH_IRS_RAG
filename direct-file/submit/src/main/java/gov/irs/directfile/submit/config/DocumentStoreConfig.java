package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "submit.documentstore")
public class DocumentStoreConfig {
    private String region;
    private String accessKey;
    private String secretKey;
    private String assumeRoleArn;
    private String assumeRoleDurationSeconds;
    private String assumeRoleSessionName;
    private String kmsWrappingKeyArn;
    private String endpoint;
    private String bucket;
    private String environmentPrefix;
}
