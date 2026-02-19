package gov.irs.boot.autoconfigure.openfeature;

import java.time.Duration;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties("openfeature-starter.s3-provider")
public record OpenFeatureConfigurationProperties(
        @NotNull String environmentPrefix, @NotBlank String bucket, @NotNull Duration expiration) {}
