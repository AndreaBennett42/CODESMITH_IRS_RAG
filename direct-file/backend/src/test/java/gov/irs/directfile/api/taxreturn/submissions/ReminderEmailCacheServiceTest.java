package gov.irs.directfile.api.taxreturn.submissions;

import java.util.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class ReminderEmailCacheServiceTest {

    ReminderEmailCacheService reminderEmailCacheService = new ReminderEmailCacheService();

    String STATE_MA_VALUE = "ma";
    String RESUBMIT_VALUE = "resubmit";

    @Test
    public void whenPut_ThenSizeEqualsNumberOfIdAddedToCache_ThenAllIdsInCache() {
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            UUID val = UUID.randomUUID();
            reminderEmailCacheService.put(val, val, STATE_MA_VALUE);
        }
        assertEquals(reminderEmailCacheService.size(STATE_MA_VALUE), 100);
        idList.forEach(id -> {
            assertEquals(id, reminderEmailCacheService.get(STATE_MA_VALUE, id));
        });
    }

    @Test
    public void whenPartialEviction_ThenCacheIsntFullyCleared() {
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i < 101; i++) {
            UUID val = UUID.randomUUID();
            idList.add(val);
            reminderEmailCacheService.put(val, val, STATE_MA_VALUE);
        }
        List<UUID> firstEvictionBatch = idList.subList(0, 50);
        List<UUID> secondEvictionBatch = idList.subList(50, 100);
        List<UUID> thirdEvictionBatch = idList.subList(100, 101);

        reminderEmailCacheService.evict(firstEvictionBatch, STATE_MA_VALUE);
        assertEquals(reminderEmailCacheService.size(STATE_MA_VALUE), 51);

        reminderEmailCacheService.evict(secondEvictionBatch, STATE_MA_VALUE);
        assertEquals(reminderEmailCacheService.size(STATE_MA_VALUE), 1);

        reminderEmailCacheService.evict(thirdEvictionBatch, STATE_MA_VALUE);
        assertEquals(reminderEmailCacheService.size(STATE_MA_VALUE), 0);
    }

    @Test
    public void whenMultipleCacheKeysExist_ThenProcessingCachePullsIdsFromAll() {
        List<UUID> maIds = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            UUID val = UUID.randomUUID();
            maIds.add(val);
            reminderEmailCacheService.put(val, val, STATE_MA_VALUE);
        }
        List<UUID> resubmitIds = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            UUID val = UUID.randomUUID();
            resubmitIds.add(val);
            reminderEmailCacheService.put(val, val, RESUBMIT_VALUE);
        }

        Map<String, List<UUID>> firstBatch =
                reminderEmailCacheService.getNextBatch().get();
        String firstKey = firstBatch.keySet().stream().findFirst().get();
        reminderEmailCacheService.evict(firstBatch.get(firstKey), firstKey);
        assertNotEquals(reminderEmailCacheService.size(RESUBMIT_VALUE), reminderEmailCacheService.size(STATE_MA_VALUE));
        assertEquals(
                reminderEmailCacheService.size(STATE_MA_VALUE) + reminderEmailCacheService.size(RESUBMIT_VALUE), 150);

        Map<String, List<UUID>> secondBatch =
                reminderEmailCacheService.getNextBatch().get();
        String secondKey = secondBatch.keySet().stream().findFirst().get();
        reminderEmailCacheService.evict(secondBatch.get(secondKey), secondKey);
        assertNotEquals(firstBatch, secondBatch);
        assertEquals(
                reminderEmailCacheService.size(STATE_MA_VALUE) + reminderEmailCacheService.size(RESUBMIT_VALUE), 100);

        Map<String, List<UUID>> thirdBatch =
                reminderEmailCacheService.getNextBatch().get();
        String thirdKey = thirdBatch.keySet().stream().findFirst().get();
        reminderEmailCacheService.evict(thirdBatch.get(thirdKey), thirdKey);
        assertNotEquals(firstBatch, thirdBatch);
        assertNotEquals(secondBatch, thirdBatch);
        assertEquals(
                reminderEmailCacheService.size(STATE_MA_VALUE) + reminderEmailCacheService.size(RESUBMIT_VALUE), 50);

        Map<String, List<UUID>> fourthBatch =
                reminderEmailCacheService.getNextBatch().get();
        String fourthKey = fourthBatch.keySet().stream().findFirst().get();
        reminderEmailCacheService.evict(fourthBatch.get(fourthKey), fourthKey);
        assertNotEquals(firstBatch, fourthBatch);
        assertNotEquals(secondBatch, fourthBatch);
        assertNotEquals(thirdBatch, fourthBatch);
        assertEquals(
                reminderEmailCacheService.size(STATE_MA_VALUE) + reminderEmailCacheService.size(RESUBMIT_VALUE), 0);
        assertEquals(reminderEmailCacheService.size(STATE_MA_VALUE), 0);
        assertEquals(reminderEmailCacheService.size(RESUBMIT_VALUE), 0);
    }
}
