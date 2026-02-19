package gov.irs.directfile.api.config;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("direct-file.status-response-body-cache")
public record StatusResponseBodyCacheProperties(Long maxItems, Duration expireAfterWrite) {}
