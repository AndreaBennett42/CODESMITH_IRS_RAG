package gov.irs.directfile.api.authentication;

import java.time.Duration;
import java.util.Optional;
import java.util.UUID;

import org.springframework.security.core.userdetails.UserDetails;

import gov.irs.directfile.api.cache.CacheService;
import gov.irs.directfile.api.config.RedisConfiguration;

public class RemoteUserDetailsCacheService implements UserDetailsCacheService {
    private final CacheService cacheService;
    private final UserDetailsCacheProperties userDetailsCacheProperties;

    public RemoteUserDetailsCacheService(
            CacheService cacheService, UserDetailsCacheProperties userDetailsCacheProperties) {
        this.cacheService = cacheService;
        this.userDetailsCacheProperties = userDetailsCacheProperties;
    }

    @Override
    public Optional<UserDetails> get(UUID userExternalId) {
        SMUserDetailsProperties properties = cacheService.get(
                RedisConfiguration.USERS_CACHE_NAME, userExternalId.toString(), SMUserDetailsProperties.class);
        if (properties == null) {
            return Optional.empty();
        }
        return Optional.of(new SMUserDetailsPrincipal(properties));
    }

    @Override
    public void put(UUID userExternalId, UserDetails userDetails) {
        SMUserDetailsProperties properties = new SMUserDetailsProperties((SMUserDetailsPrincipal) userDetails);

        Duration expireAfterWrite = userDetailsCacheProperties.expireAfterWrite();
        if (expireAfterWrite != null) {
            cacheService.set(
                    RedisConfiguration.USERS_CACHE_NAME, userExternalId.toString(), properties, expireAfterWrite);
        } else {
            cacheService.set(RedisConfiguration.USERS_CACHE_NAME, userExternalId.toString(), properties);
        }
    }

    @Override
    public void clear() {
        cacheService.clearCache(RedisConfiguration.USERS_CACHE_NAME);
    }
}
