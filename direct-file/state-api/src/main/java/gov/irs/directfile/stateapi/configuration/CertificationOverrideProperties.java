package gov.irs.directfile.stateapi.configuration;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@RequiredArgsConstructor
@Getter
@Validated
@ConfigurationProperties(prefix = "direct-file")
public class CertificationOverrideProperties {
    private final String certLocationOverride;
}
