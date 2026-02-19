package gov.irs.directfile.stateapi.configuration;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@AllArgsConstructor
@ConfigurationProperties(prefix = "feature-flags")
public class FeatureFlagsConfigurationProperties {
    private final ExportReturnFlags exportReturn;

    public interface Toggleable {
        boolean isEnabled();
    }

    @Getter
    @AllArgsConstructor
    public static class ExportReturnFlags implements Toggleable {
        private boolean enabled;
    }
}
