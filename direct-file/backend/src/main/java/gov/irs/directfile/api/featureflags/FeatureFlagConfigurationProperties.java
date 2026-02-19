package gov.irs.directfile.api.featureflags;

import java.time.Duration;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

// TODO: evaluate changing prefix since this is configuration for feature flags rather than S3
@ConfigurationProperties(prefix = "direct-file.aws.s3")
@Getter
@AllArgsConstructor
public class FeatureFlagConfigurationProperties {
    private @NotNull String environmentPrefix;
    private @NotNull String featureFlagsBucket;
    private @NotNull String featureFlagsObject;
    private @NotNull Duration featureFlagsExpiration;
}
