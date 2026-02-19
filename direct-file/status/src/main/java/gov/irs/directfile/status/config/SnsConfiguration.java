package gov.irs.directfile.status.config;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.hibernate.validator.constraints.URL;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@AllArgsConstructor
@Getter
@ConfigurationProperties("status.sns")
public class SnsConfiguration {
    @NotBlank
    @URL
    private final String endpoint;

    @NotBlank
    private final String statusChangeTopicArn;

    @NotBlank
    private final String region;

    @NotBlank
    private final String accessKey;

    @NotBlank
    private final String secretKey;

    private final boolean statusChangePublishEnabled;
}
