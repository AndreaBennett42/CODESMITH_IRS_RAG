package gov.irs.directfile.api.taxreturn;

import java.time.Duration;
import java.util.Date;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.cache.CacheService;
import gov.irs.directfile.api.config.RedisConfiguration;
import gov.irs.directfile.api.config.StatusResponseBodyCacheProperties;
import gov.irs.directfile.api.taxreturn.dto.Status;
import gov.irs.directfile.api.taxreturn.dto.StatusResponseBody;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RemoteStatusResponseBodyCacheServiceTest {
    RemoteStatusResponseBodyCacheService remoteStatusResponseBodyCacheService;

    @Mock
    CacheService cacheService;

    StatusResponseBodyCacheProperties properties;

    private static final String SUBMISSION_ID = "138913903abc";

    @BeforeEach
    void test() {
        properties = new StatusResponseBodyCacheProperties(5L, Duration.ofSeconds(5));
        remoteStatusResponseBodyCacheService = new RemoteStatusResponseBodyCacheService(cacheService, properties);
    }

    @Test
    void get_callsCacheServiceGetHasHit() {
        Date now = new Date();
        StatusResponseBody statusResponseBodyValue =
                new StatusResponseBody(Status.Accepted, "status.accepted", List.of(), now);

        when(cacheService.get(RedisConfiguration.STATUS_CACHE_NAME, SUBMISSION_ID, StatusResponseBody.class))
                .thenReturn(statusResponseBodyValue);

        Optional<StatusResponseBody> statusResponseBody = remoteStatusResponseBodyCacheService.get(SUBMISSION_ID);

        assertTrue(statusResponseBody.isPresent());
        StatusResponseBody expected = new StatusResponseBody(Status.Accepted, "status.accepted", List.of(), now);
        assertEquals(expected, statusResponseBody.get());
    }

    @Test
    void get_callsCacheServiceGetHasMiss() {
        when(cacheService.get(RedisConfiguration.STATUS_CACHE_NAME, SUBMISSION_ID, StatusResponseBody.class))
                .thenReturn(null);

        Optional<StatusResponseBody> statusResponseBody = remoteStatusResponseBodyCacheService.get(SUBMISSION_ID);

        assertTrue(statusResponseBody.isEmpty());
    }

    @Test
    void put_callsCacheServiceSetWithExpiration() {
        StatusResponseBody statusResponseBodyValue =
                new StatusResponseBody(Status.Accepted, "status.accepted", List.of(), new Date());

        remoteStatusResponseBodyCacheService.put(SUBMISSION_ID, statusResponseBodyValue);

        verify(cacheService, times(1))
                .set(
                        RedisConfiguration.STATUS_CACHE_NAME,
                        SUBMISSION_ID,
                        statusResponseBodyValue,
                        properties.expireAfterWrite());
    }

    @Test
    void put_callsCacheServiceSetNoExpiration() {
        properties = new StatusResponseBodyCacheProperties(5L, null);
        remoteStatusResponseBodyCacheService = new RemoteStatusResponseBodyCacheService(cacheService, properties);

        StatusResponseBody statusResponseBodyValue =
                new StatusResponseBody(Status.Accepted, "status.accepted", List.of(), new Date());

        remoteStatusResponseBodyCacheService.put(SUBMISSION_ID, statusResponseBodyValue);

        verify(cacheService, times(1))
                .set(RedisConfiguration.STATUS_CACHE_NAME, SUBMISSION_ID, statusResponseBodyValue);
    }

    @Test
    void clearKey_callsCacheServiceClearKey() {
        remoteStatusResponseBodyCacheService.clearKey(SUBMISSION_ID);

        verify(cacheService, times(1)).clearKey(RedisConfiguration.STATUS_CACHE_NAME, SUBMISSION_ID);
    }

    @Test
    void clear_callsCacheServiceClear() {
        remoteStatusResponseBodyCacheService.clear();

        verify(cacheService, times(1)).clearCache(RedisConfiguration.STATUS_CACHE_NAME);
    }
}
