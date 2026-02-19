package gov.irs.directfile.stateapi.configuration;

import java.util.List;
import java.util.concurrent.TimeUnit;

import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/***
 * Configures the cache used by Spring @Cacheable methods, such as those found in the CachedDataService
 */
@Configuration
public class CacheConfiguration {

    @Value("${spring.cache.TTL-minutes: 120}")
    private long cacheTTL;

    @Bean
    public Caffeine<Object, Object> caffeineConfig() {
        return Caffeine.newBuilder()
                .expireAfterWrite(cacheTTL, TimeUnit.MINUTES)
                .maximumSize(100);
    }

    @Bean
    public CacheManager cacheManager(Caffeine<Object, Object> caffeine) {
        var caffeineCacheManager = new CaffeineCacheManager();
        caffeineCacheManager.setCaffeine(caffeine);
        caffeineCacheManager.setCacheNames(List.of("publicKeyCache", "stateProfileCache"));
        caffeineCacheManager.setAsyncCacheMode(true);
        return caffeineCacheManager;
    }
}
