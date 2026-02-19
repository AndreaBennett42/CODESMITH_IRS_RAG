package gov.irs.directfile.api.authorization.config;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.stereotype.Component;

@Getter
@Component
public class FeatureFlagConfigurationProperties {
    @NotNull private final Allowlist allowlist;

    @NotNull private final OpenEnrollment openEnrollment;

    public FeatureFlagConfigurationProperties() {
        this.allowlist = new Allowlist(false, "key", "allowlist.csv");
        this.openEnrollment = new OpenEnrollment(true);
    }

    @Getter
    @AllArgsConstructor
    public static class OpenEnrollment {
        @NotNull private final boolean enabled;
    }

    public record Allowlist(@NotNull boolean enabled, @NotNull String hexKey, @NotNull String objectKey) {}
}
