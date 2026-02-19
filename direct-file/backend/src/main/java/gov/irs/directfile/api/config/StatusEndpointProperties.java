package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.hibernate.validator.constraints.URL;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "direct-file.status-endpoint")
@Validated
public class StatusEndpointProperties {
    @NotBlank
    @URL
    private String statusEndpointURI;

    @NotBlank
    @URL
    private String rejectionCodesEndpointURI;
}
