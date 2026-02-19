package gov.irs.directfile.api.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "direct-file.submit-endpoint")
public class SubmitEndpointProperties {
    private String submitEndpointURI;
}
