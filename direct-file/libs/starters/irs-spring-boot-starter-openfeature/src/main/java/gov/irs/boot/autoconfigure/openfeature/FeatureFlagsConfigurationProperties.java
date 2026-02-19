package gov.irs.boot.autoconfigure.openfeature;

import java.util.Map;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties
public record FeatureFlagsConfigurationProperties(Map<String, FeatureFlag> featureFlags) {

    @Getter
    @RequiredArgsConstructor
    public static class FeatureFlag {
        private final Map<String, Object> variants;
        private final String defaultVariant;
    }
}
