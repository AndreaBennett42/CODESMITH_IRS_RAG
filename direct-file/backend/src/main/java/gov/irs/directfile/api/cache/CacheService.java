package gov.irs.directfile.api.cache;

import java.time.Duration;
import java.util.Set;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
@Slf4j
@AllArgsConstructor
public class CacheService {
    private RedisTemplate<String, String> redisTemplate;
    private ObjectMapper objectMapper;

    public static final String KEY_SEPARATOR = ":";
    public static final String KEY_GLOB = "*";

    private String makeCacheKeyString(String cacheName, String key) {
        return String.join(KEY_SEPARATOR, cacheName, key);
    }

    public <V> void set(String cacheName, String key, V value) {
        set(cacheName, key, value, null);
    }

    public <V> void set(String cacheName, String key, V value, Duration timeout) {
        String remoteCacheKeyString = makeCacheKeyString(cacheName, key);
        try {
            String serializedValue = objectMapper.writeValueAsString(value);
            if (timeout != null) {
                log.info("Setting cache key {} with timeout {}", remoteCacheKeyString, timeout);
                redisTemplate.opsForValue().set(remoteCacheKeyString, serializedValue, timeout);
            } else {
                log.info("Setting cache key {}", remoteCacheKeyString);
                redisTemplate.opsForValue().set(remoteCacheKeyString, serializedValue);
            }
        } catch (Exception e) {
            log.error(
                    "Unable to set data from Redis. {}: {} {} {}",
                    e.getClass(),
                    e.getMessage(),
                    e.getCause(),
                    e.getStackTrace());
        }
    }

    public <V> V get(String cacheName, String key, Class<V> clazz) {
        String remoteCacheKeyString = makeCacheKeyString(cacheName, key);
        try {
            log.info("Getting cache key {}", remoteCacheKeyString);
            String serializedValue = redisTemplate.opsForValue().get(remoteCacheKeyString);
            return (serializedValue != null) ? objectMapper.readValue(serializedValue, clazz) : null;
        } catch (Exception e) {
            log.error(
                    "Unable to fetch data from Redis. {}: {} {} {}",
                    e.getClass(),
                    e.getMessage(),
                    e.getCause(),
                    e.getStackTrace());
            return null;
        }
    }

    public void clearKey(String cacheName, String key) {
        String remoteCacheKeyString = makeCacheKeyString(cacheName, key);
        try {
            log.info("Clearing cache key {}", remoteCacheKeyString);
            redisTemplate.delete(remoteCacheKeyString);
        } catch (Exception e) {
            log.error("Unable to clear key from Redis. {}: {}", e.getClass(), e.getMessage());
        }
    }

    public void clearCache(String cacheName) {
        String keyPatternToDelete = makeCacheKeyString(cacheName, KEY_GLOB);
        try {
            log.info("Clearing cache pattern {}", keyPatternToDelete);
            Set<String> keys = redisTemplate.keys(keyPatternToDelete);
            if (keys != null) {
                redisTemplate.delete(keys);
            }
        } catch (Exception e) {
            log.error(
                    "Unable to clear data from Redis. {}: {} {} {}",
                    e.getClass(),
                    e.getMessage(),
                    e.getCause(),
                    e.getStackTrace());
        }
    }
}
