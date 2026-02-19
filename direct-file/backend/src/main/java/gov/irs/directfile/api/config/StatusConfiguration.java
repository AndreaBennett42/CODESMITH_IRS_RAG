package gov.irs.directfile.api.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

import gov.irs.directfile.api.cache.CacheService;
import gov.irs.directfile.api.taxreturn.LocalStatusResponseBodyCacheService;
import gov.irs.directfile.api.taxreturn.RemoteStatusResponseBodyCacheService;
import gov.irs.directfile.api.taxreturn.StatusResponseBodyCacheService;

@Configuration
@EnableConfigurationProperties(StatusResponseBodyCacheProperties.class)
public class StatusConfiguration {
    @Bean
    @Profile("!" + BeanProfiles.ENABLE_REMOTE_CACHE) // use in memory cache
    public StatusResponseBodyCacheService localStatusResponseBodyCacheService(
            StatusResponseBodyCacheProperties statusResponseBodyCacheProperties) {
        return new LocalStatusResponseBodyCacheService(statusResponseBodyCacheProperties);
    }

    @Bean
    @Profile(BeanProfiles.ENABLE_REMOTE_CACHE) // use remote cache for local development
    public StatusResponseBodyCacheService remoteStatusResponseBodyCacheService(
            CacheService cacheService, StatusResponseBodyCacheProperties statusResponseBodyCacheProperties) {
        return new RemoteStatusResponseBodyCacheService(cacheService, statusResponseBodyCacheProperties);
    }
}
