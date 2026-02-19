package gov.irs.directfile.api.taxreturn;

import java.time.Duration;
import java.util.Optional;

import gov.irs.directfile.api.cache.CacheService;
import gov.irs.directfile.api.config.RedisConfiguration;
import gov.irs.directfile.api.config.StatusResponseBodyCacheProperties;
import gov.irs.directfile.api.taxreturn.dto.StatusResponseBody;

public class RemoteStatusResponseBodyCacheService implements StatusResponseBodyCacheService {
    private final CacheService cacheService;
    private final StatusResponseBodyCacheProperties statusResponseBodyCacheProperties;

    public RemoteStatusResponseBodyCacheService(
            CacheService cacheService, StatusResponseBodyCacheProperties statusResponseBodyCacheProperties) {
        this.cacheService = cacheService;
        this.statusResponseBodyCacheProperties = statusResponseBodyCacheProperties;
    }

    @Override
    public Optional<StatusResponseBody> get(String submissionId) {
        return Optional.ofNullable(
                cacheService.get(RedisConfiguration.STATUS_CACHE_NAME, submissionId, StatusResponseBody.class));
    }

    @Override
    public void put(String submissionId, StatusResponseBody statusResponseBody) {
        Duration expireAfterWrite = statusResponseBodyCacheProperties.expireAfterWrite();
        if (expireAfterWrite != null) {
            cacheService.set(RedisConfiguration.STATUS_CACHE_NAME, submissionId, statusResponseBody, expireAfterWrite);
        } else {
            cacheService.set(RedisConfiguration.STATUS_CACHE_NAME, submissionId, statusResponseBody);
        }
    }

    @Override
    public void clearKey(String submissionId) {
        cacheService.clearKey(RedisConfiguration.STATUS_CACHE_NAME, submissionId);
    }

    @Override
    public void clear() {
        cacheService.clearCache(RedisConfiguration.STATUS_CACHE_NAME);
    }
}
