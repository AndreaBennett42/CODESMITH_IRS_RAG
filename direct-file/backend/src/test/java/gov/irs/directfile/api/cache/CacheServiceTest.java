package gov.irs.directfile.api.cache;

import java.time.Duration;
import java.util.Set;
import java.util.stream.Stream;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CacheServiceTest {

    CacheService cacheService;

    @Mock
    RedisTemplate<String, String> redisTemplate;

    @Mock
    ValueOperations<String, String> valueOperations;

    @BeforeEach
    void test() {
        cacheService = new CacheService(redisTemplate, new ObjectMapper());
    }

    @Test
    void set_callsRedisTemplateSet() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);

        cacheService.set("cacheName", "key1", "value1");

        verify(valueOperations, times(1)).set(eq("cacheName" + CacheService.KEY_SEPARATOR + "key1"), eq("\"value1\""));
    }

    @Test
    void set_callsRedisTemplateSetWithTimeout() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);

        Duration duration = Duration.ofSeconds(5);
        cacheService.set("cacheName", "key1", "value1", duration);

        verify(valueOperations, times(1))
                .set(eq("cacheName" + CacheService.KEY_SEPARATOR + "key1"), eq("\"value1\""), eq(duration));
    }

    private static Stream<Arguments> getCacheGetParameters() {
        return Stream.of(Arguments.of("\"value1\"", "value1"), Arguments.of(null, null), Arguments.of("bad", null));
    }

    @ParameterizedTest
    @MethodSource("getCacheGetParameters")
    void get_callsRedisTemplateGet(String mockReturnValue, String expectedReturnValue) {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("cacheName" + CacheService.KEY_SEPARATOR + "key1"))
                .thenReturn(mockReturnValue);

        String value = cacheService.get("cacheName", "key1", String.class);

        verify(valueOperations, times(1)).get(eq("cacheName" + CacheService.KEY_SEPARATOR + "key1"));
        assertEquals(value, expectedReturnValue);
    }

    @Test
    void clearKey_callsRedisTemplateDelete() {
        cacheService.clearKey("cacheName", "key1");

        verify(redisTemplate, times(1)).delete(eq("cacheName" + CacheService.KEY_SEPARATOR + "key1"));
    }

    @Test
    void clearCache_whenCacheHasKeys_callsRedisTemplateHasKey() {
        Set<String> keys = Set.of("key1", "key2");

        when(redisTemplate.keys("cacheName" + CacheService.KEY_SEPARATOR + CacheService.KEY_GLOB))
                .thenReturn(keys);
        cacheService.clearCache("cacheName");

        verify(redisTemplate, times(1)).delete(keys);
    }

    @Test
    void clearCache_whenCacheHasNullKeys_callsRedisTemplateHasKey() {
        Set<String> keys = Set.of("key1", "key2");

        when(redisTemplate.keys("cacheName" + CacheService.KEY_SEPARATOR + CacheService.KEY_GLOB))
                .thenReturn(null);
        cacheService.clearCache("cacheName");

        verify(redisTemplate, never()).delete(keys);
    }
}
