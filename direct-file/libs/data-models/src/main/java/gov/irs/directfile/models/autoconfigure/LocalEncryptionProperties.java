package gov.irs.directfile.models.autoconfigure;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "direct-file.local-encryption")
@Getter
@AllArgsConstructor
public class LocalEncryptionProperties {
    public String localWrappingKey;
}
