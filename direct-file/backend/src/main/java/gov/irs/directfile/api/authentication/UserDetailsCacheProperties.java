package gov.irs.directfile.api.authentication;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("direct-file.user-details-cache")
public record UserDetailsCacheProperties(Long maxItems, Duration expireAfterWrite) {}
