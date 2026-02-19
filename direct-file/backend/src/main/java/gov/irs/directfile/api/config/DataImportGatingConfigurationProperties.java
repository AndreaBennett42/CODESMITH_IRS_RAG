package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "direct-file.data-import-gating")
@Getter
@AllArgsConstructor
public class DataImportGatingConfigurationProperties {
    @NotNull private final Allowlist allowlist;

    public record Allowlist(@NotNull boolean enabled, @NotNull String hexKey, @NotNull String objectKey) {}
}
