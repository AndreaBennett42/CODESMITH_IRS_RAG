package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "direct-file")
@Getter
@AllArgsConstructor
public class DirectFileConfigurationProperties {
    @NotNull private String apiVersion;
}
