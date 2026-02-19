package gov.irs.directfile.stateapi.configuration;

import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@RequiredArgsConstructor
@Getter
@Setter
@Validated
@ConfigurationProperties(prefix = "direct-file")
public class DirectFileEndpointProperties {
    @NonNull private String backendUrl;

    @NonNull private String backendContextPath;

    @NonNull private String backendApiVersion;
}
