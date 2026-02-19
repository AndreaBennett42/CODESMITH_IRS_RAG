package gov.irs.directfile.api.taxreturn;

import java.time.Duration;
import java.util.Date;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.api.config.StatusResponseBodyCacheProperties;
import gov.irs.directfile.api.taxreturn.dto.Status;
import gov.irs.directfile.api.taxreturn.dto.StatusResponseBody;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LocalStatusResponseBodyCacheServiceTest {
    // basic test of functionality, anything more complex should be trusted to the backing guava cache
    private static final String SUBMISSION_ID1 = "303981309abc";
    private static final String SUBMISSION_ID2 = "173838390xyz";

    private static final StatusResponseBody statusResponseBody1 =
            new StatusResponseBody(Status.Accepted, "status.accepted", List.of(), new Date());
    private static final StatusResponseBody statusResponseBody2 =
            new StatusResponseBody(Status.Error, "status.error", List.of(), new Date());

    private StatusResponseBodyCacheService statusResponseBodyCacheService;

    @BeforeEach
    void setup() {
        StatusResponseBodyCacheProperties properties = new StatusResponseBodyCacheProperties(5L, Duration.ofSeconds(5));
        statusResponseBodyCacheService = new LocalStatusResponseBodyCacheService(properties);
        statusResponseBodyCacheService.clear();
    }

    @Test
    void givenCacheService_whenGetPutGet_thenItemIsCached() {
        // when
        Optional<StatusResponseBody> get1StatusResponseBody = statusResponseBodyCacheService.get(SUBMISSION_ID1);
        statusResponseBodyCacheService.put(SUBMISSION_ID1, statusResponseBody1);
        Optional<StatusResponseBody> get2StatusResponseBody = statusResponseBodyCacheService.get(SUBMISSION_ID1);

        // then
        assertTrue(get1StatusResponseBody.isEmpty());
        assertEquals(statusResponseBody1, get2StatusResponseBody.get());
    }

    @Test
    void givenCacheService_whenPutThenClearKey_thenItemIsDeleted() {
        // when
        statusResponseBodyCacheService.put(SUBMISSION_ID1, statusResponseBody1);
        Optional<StatusResponseBody> get1StatusResponseBody = statusResponseBodyCacheService.get(SUBMISSION_ID1);
        statusResponseBodyCacheService.clearKey(SUBMISSION_ID1);
        Optional<StatusResponseBody> get2StatusResponseBody = statusResponseBodyCacheService.get(SUBMISSION_ID1);

        // then
        assertEquals(statusResponseBody1, get1StatusResponseBody.get());
        assertTrue(get2StatusResponseBody.isEmpty());
    }

    @Test
    void givenCacheService_whenPutMultipleThenClearAll_thenAllItemsAreDeleted() {
        // when
        statusResponseBodyCacheService.put(SUBMISSION_ID1, statusResponseBody1);
        statusResponseBodyCacheService.put(SUBMISSION_ID2, statusResponseBody2);
        Optional<StatusResponseBody> get1StatusResponseBody1 = statusResponseBodyCacheService.get(SUBMISSION_ID1);
        Optional<StatusResponseBody> get1StatusResponseBody2 = statusResponseBodyCacheService.get(SUBMISSION_ID2);
        statusResponseBodyCacheService.clear();
        Optional<StatusResponseBody> get2StatusResponseBody1 = statusResponseBodyCacheService.get(SUBMISSION_ID1);
        Optional<StatusResponseBody> get2StatusResponseBody2 = statusResponseBodyCacheService.get(SUBMISSION_ID2);

        // then
        assertEquals(statusResponseBody1, get1StatusResponseBody1.get());
        assertEquals(statusResponseBody2, get1StatusResponseBody2.get());
        assertTrue(get2StatusResponseBody1.isEmpty());
        assertTrue(get2StatusResponseBody2.isEmpty());
    }
}
