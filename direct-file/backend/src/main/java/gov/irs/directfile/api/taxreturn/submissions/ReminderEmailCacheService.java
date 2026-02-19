package gov.irs.directfile.api.taxreturn.submissions;

import java.util.*;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class ReminderEmailCacheService {
    private final Cache<String, Map<String, String>> reminderEmailCache;

    public ReminderEmailCacheService() {
        this.reminderEmailCache = CacheBuilder.newBuilder().maximumSize(50000).build();
    }

    public void put(UUID idKey, UUID idVal, String reminderCategory) {
        Map<String, String> categoryMap = getOrCreateCategoryMap(reminderCategory);
        categoryMap.put(String.valueOf(idKey), String.valueOf(idVal));
    }

    public UUID get(String reminderCategory, UUID taxReturnId) {
        Map<String, String> categoryMap = getCategoryMap(reminderCategory);
        return UUID.fromString(categoryMap.get(taxReturnId.toString()));
    }

    public int size(String reminderCategory) {
        Map<String, String> categoryMap = getCategoryMap(reminderCategory);
        int size;
        if (categoryMap == null) {
            log.info("ReminderEmailCacheService has cache size of 0");
            return 0;
        } else {
            size = categoryMap.size();
            log.info("ReminderEmailCacheService has cache size of {}", size);
            return size;
        }
    }

    public void evict(UUID taxReturnId, String reminderCategory) {
        getCategoryMap(reminderCategory).remove(taxReturnId.toString());
    }

    public void evict(List<UUID> taxReturnIds, String reminderCategory) {
        taxReturnIds.forEach(id -> evict(id, reminderCategory));
        Map<String, String> updatedMap = getCategoryMap(reminderCategory);
        if (updatedMap.isEmpty()) {
            this.reminderEmailCache.asMap().remove(reminderCategory);
        }
    }

    public Map<String, String> getCategoryMap(String reminderCategory) {
        return this.reminderEmailCache.getIfPresent(reminderCategory);
    }

    public Map<String, String> getOrCreateCategoryMap(String reminderCategory) {
        Map<String, String> categoryMap = this.reminderEmailCache.getIfPresent(reminderCategory);
        if (categoryMap == null) {
            this.reminderEmailCache.put(reminderCategory, new HashMap<>());
            return this.reminderEmailCache.getIfPresent(reminderCategory);
        } else {
            return categoryMap;
        }
    }

    public Optional<Map<String, List<UUID>>> getNextBatch() {
        Optional<String> optKey =
                this.reminderEmailCache.asMap().keySet().stream().findFirst();
        if (optKey.isPresent()) {
            String key = optKey.get();
            List<String> idsInCache = getCategoryMap(key).values().stream().toList();
            List<UUID> taxReturnIds = new ArrayList<>();
            Map<String, List<UUID>> categoryTaxReturnIdsMap = new HashMap<>();
            for (int i = 0; i < 50; i++) {
                try {
                    String nextId = idsInCache.get(i);
                    taxReturnIds.add(UUID.fromString(nextId));
                } catch (IndexOutOfBoundsException e) {
                    break;
                }
            }
            categoryTaxReturnIdsMap.put(key, taxReturnIds);
            return Optional.of(categoryTaxReturnIdsMap);
        } else {
            return Optional.empty();
        }
    }
}
