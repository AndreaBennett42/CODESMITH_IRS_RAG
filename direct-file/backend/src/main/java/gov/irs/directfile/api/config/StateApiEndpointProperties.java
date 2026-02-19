package gov.irs.directfile.api.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "direct-file.state-api")
public class StateApiEndpointProperties {
    private String baseUrl;
    private String v2AuthTokenPath;
    private String version;
}
