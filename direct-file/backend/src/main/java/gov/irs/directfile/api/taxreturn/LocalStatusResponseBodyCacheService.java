package gov.irs.directfile.api.taxreturn;

import java.time.Duration;
import java.util.Optional;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import gov.irs.directfile.api.config.StatusResponseBodyCacheProperties;
import gov.irs.directfile.api.taxreturn.dto.StatusResponseBody;

@EnableConfigurationProperties(StatusResponseBodyCacheProperties.class)
public class LocalStatusResponseBodyCacheService implements StatusResponseBodyCacheService {
    private final Cache<String, StatusResponseBody> statusResponseBodyCache;

    public LocalStatusResponseBodyCacheService(StatusResponseBodyCacheProperties statusResponseBodyCacheProperties) {
        CacheBuilder<Object, Object> builder = CacheBuilder.newBuilder();

        Long maxItems = statusResponseBodyCacheProperties.maxItems();
        if (maxItems != null) {
            builder.maximumSize(maxItems);
        }

        Duration expireAfterWrite = statusResponseBodyCacheProperties.expireAfterWrite();
        if (expireAfterWrite != null) {
            builder.expireAfterWrite(expireAfterWrite);
        }

        this.statusResponseBodyCache = builder.build();
    }

    @Override
    public Optional<StatusResponseBody> get(String submissionId) {
        return Optional.ofNullable(statusResponseBodyCache.getIfPresent(submissionId));
    }

    @Override
    public void put(String submissionId, StatusResponseBody statusResponseBody) {
        statusResponseBodyCache.put(submissionId, statusResponseBody);
    }

    @Override
    public void clearKey(String submissionId) {
        statusResponseBodyCache.invalidate(submissionId);
    }

    @Override
    public void clear() {
        statusResponseBodyCache.invalidateAll();
    }
}
