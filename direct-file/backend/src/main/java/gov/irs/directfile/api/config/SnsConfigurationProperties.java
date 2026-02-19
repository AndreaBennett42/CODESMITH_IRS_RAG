package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.hibernate.validator.constraints.URL;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties("direct-file.aws.sns")
@Getter
@AllArgsConstructor
public class SnsConfigurationProperties {
    @NotBlank
    @URL
    private final String endpoint;

    @NotBlank
    private final String submissionConfirmationTopicArn;

    private final boolean submissionConfirmationPublishEnabled;

    @NotBlank
    private final String region;
}
