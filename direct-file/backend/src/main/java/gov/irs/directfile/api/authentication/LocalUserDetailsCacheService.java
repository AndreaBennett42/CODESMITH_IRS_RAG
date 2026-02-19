package gov.irs.directfile.api.authentication;

import java.time.Duration;
import java.util.Optional;
import java.util.UUID;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import org.springframework.security.core.userdetails.UserDetails;

public class LocalUserDetailsCacheService implements UserDetailsCacheService {
    private final Cache<UUID, UserDetails> userDetailsCache;

    public LocalUserDetailsCacheService(UserDetailsCacheProperties userDetailsCacheProperties) {
        CacheBuilder<Object, Object> builder = CacheBuilder.newBuilder();

        Long maxItems = userDetailsCacheProperties.maxItems();
        if (maxItems != null) {
            builder.maximumSize(maxItems);
        }

        Duration expireAfterWrite = userDetailsCacheProperties.expireAfterWrite();
        if (expireAfterWrite != null) {
            builder.expireAfterWrite(expireAfterWrite);
        }

        this.userDetailsCache = builder.build();
    }

    @Override
    public Optional<UserDetails> get(UUID userExternalId) {
        return Optional.ofNullable(userDetailsCache.getIfPresent(userExternalId));
    }

    @Override
    public void put(UUID userExternalId, UserDetails userDetails) {
        userDetailsCache.put(userExternalId, userDetails);
    }

    @Override
    public void clear() {
        userDetailsCache.invalidateAll();
    }
}
