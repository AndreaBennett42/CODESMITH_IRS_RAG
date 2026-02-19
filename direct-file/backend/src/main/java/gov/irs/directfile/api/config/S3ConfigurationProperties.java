package gov.irs.directfile.api.config;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties("direct-file.aws")
@Getter
@AllArgsConstructor
public class S3ConfigurationProperties {

    @NotBlank
    private final String region;

    @NotNull private final Credentials credentials;

    @NotNull private final S3 s3;

    @Getter
    @AllArgsConstructor
    public static class Credentials {
        @NotBlank
        private final String accessKey;

        @NotBlank
        private final String secretKey;
    }

    @Getter
    @AllArgsConstructor
    public static class S3 {
        @NotBlank
        private final String endpoint;

        @NotBlank
        private final String assumeRoleArn;

        @NotNull @Min(value = 0)
        private final int assumeRoleDurationSeconds;

        @NotBlank
        private final String assumeRoleSessionName;

        private final String kmsWrappingKeyArn;

        @NotBlank
        private final String bucket;

        @NotBlank
        private final String operationsJobsBucket;

        private final String environmentPrefix;
    }
}
