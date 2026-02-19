package gov.irs.directfile.api.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties("aws")
@Getter
@AllArgsConstructor
public class AwsConfigurationProperties {
    private final String kmsEndpoint;
}
