package gov.irs.directfile.stateapi.configuration;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Getter
@Setter
@NoArgsConstructor
@Validated
@ConfigurationProperties(prefix = "aws.s3")
public class S3ConfigurationProperties {
    private String accessKey;
    private String secretKey;
    private String region;
    private String assumeRoleArn;
    private String assumeRoleDurationSeconds;
    private String assumeRoleSessionName;
    private String endPoint;
    private String certBucketName;
    private String taxReturnXmlBucketName;
    private String prefix;
}
