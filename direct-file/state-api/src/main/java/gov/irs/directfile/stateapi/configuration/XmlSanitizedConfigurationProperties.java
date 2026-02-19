package gov.irs.directfile.stateapi.configuration;

import java.util.List;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@RequiredArgsConstructor
@ConfigurationProperties(prefix = "xml-sanitized")
public class XmlSanitizedConfigurationProperties {
    private final List<String> allowedHeaders;
    private final List<String> excludedTags;
}
