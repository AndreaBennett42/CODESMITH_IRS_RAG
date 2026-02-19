package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@ConfigurationProperties(prefix = "direct-file.feature-flags.state-api")
@Getter
@AllArgsConstructor
@Validated
public class StateApiFeatureFlagProperties {
    @NotNull private final ExportedFacts exportedFacts;

    public record ExportedFacts(@NotNull boolean enabled) {}
}
