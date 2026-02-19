package gov.irs.directfile.api.config;

import java.time.Duration;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "direct-file.aws.s3")
@Getter
@AllArgsConstructor
public class DataImportGatingS3Properties {
    private final String environmentPrefix;
    private final @NotNull String dataImportGatingBucket;
    private final @NotNull String dataImportGatingObject;
    private final @NotNull Duration dataImportGatingExpiration;
}
