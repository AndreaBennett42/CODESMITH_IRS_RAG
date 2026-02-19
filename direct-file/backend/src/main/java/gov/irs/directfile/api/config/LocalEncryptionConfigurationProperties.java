package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties("direct-file.local-encryption")
@Getter
@AllArgsConstructor
public class LocalEncryptionConfigurationProperties {
    @NotBlank
    private final String localWrappingKey;
}
