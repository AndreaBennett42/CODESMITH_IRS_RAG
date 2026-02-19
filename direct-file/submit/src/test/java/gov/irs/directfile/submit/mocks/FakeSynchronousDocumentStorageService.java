package gov.irs.directfile.submit.mocks;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.time.Clock;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.domain.DocumentStoreResource;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

public class FakeSynchronousDocumentStorageService implements ISynchronousDocumentStoreService {
    Map<String, String> prefixToContent = new HashMap<>();
    Map<String, String> prefixToUniqueId = new HashMap<>();
    Map<String, Instant> prefixToLastModified = new HashMap<>();
    // Get time since the current clock
    private final Clock clock;
    private final Instant initializationTime = Instant.now();

    public FakeSynchronousDocumentStorageService() {
        this.clock = new MutableTestClock();
    }

    public FakeSynchronousDocumentStorageService(Clock clock) {
        this.clock = clock;
    }

    @Override
    public String write(String objectKey, InputStream payloadStream) throws IOException {
        String contentString =
                new BufferedReader(new InputStreamReader(payloadStream)).lines().collect(Collectors.joining("\n"));

        prefixToContent.put(objectKey, contentString);
        prefixToLastModified.put(
                objectKey, Instant.now(clock).plus(millisecondsSinceInitialization(), ChronoUnit.MILLIS));

        if (!prefixToUniqueId.containsKey(objectKey)) {
            String uniqueId = UUID.randomUUID().toString();
            prefixToUniqueId.put(objectKey, uniqueId);
        }
        return prefixToUniqueId.get(objectKey);
    }

    @Override
    public String write(String objectKey, String content) {
        prefixToContent.put(objectKey, content);
        prefixToLastModified.put(
                objectKey, Instant.now(clock).plus(millisecondsSinceInitialization(), ChronoUnit.MILLIS));

        if (!prefixToUniqueId.containsKey(objectKey)) {
            String uniqueId = UUID.randomUUID().toString();
            prefixToUniqueId.put(objectKey, uniqueId);
        }
        return prefixToUniqueId.get(objectKey);
    }

    @Override
    public Optional<String> getMostRecentFolderForPrefix(String prefix) {
        List<String> keysWithPrefix = prefixToContent.keySet().stream()
                .filter(key -> key.startsWith(prefix))
                .toList();

        if (keysWithPrefix.isEmpty()) {
            return Optional.empty();
        } else {
            String mostRecentlyModifiedObject = keysWithPrefix.stream()
                    .max(Comparator.comparing(key -> prefixToLastModified.get(key)))
                    .get();
            int lengthOfBatchNumber =
                    mostRecentlyModifiedObject.substring(prefix.length()).indexOf("/");

            return Optional.of(mostRecentlyModifiedObject.substring(0, prefix.length() + lengthOfBatchNumber + 1));
        }
    }

    @Override
    public List<String> getSubFolders(String objectKey) {

        List<String> result = prefixToContent.keySet().stream()
                .filter(key -> key.startsWith(objectKey))
                .map(key -> getNextFolderAfterKey(objectKey, key))
                .distinct()
                .collect(Collectors.toList());
        return result;
    }

    @Override
    public Optional<DocumentStoreResource> getLeastRecentModifiedResourceForPrefix(String s) {
        List<String> result = prefixToContent.keySet().stream()
                .filter(key -> key.startsWith(s))
                .toList();

        return result.stream()
                .map(key -> new DocumentStoreResource(key, prefixToUniqueId.get(key), prefixToLastModified.get(key)))
                .min(Comparator.comparing(DocumentStoreResource::getLastModified));
    }

    @Override
    public String getObjectAsString(String objectKey) throws IOException {
        return this.prefixToContent.get(objectKey);
    }

    @Override
    public List<DocumentStoreResource> getObjectKeys(String prefix) {
        return prefixToContent.keySet().stream()
                .filter(key -> key.startsWith(prefix))
                .map(key -> new DocumentStoreResource(key, prefixToUniqueId.get(key), prefixToLastModified.get(key)))
                .collect(Collectors.toList());
    }

    @Override
    public void Setup(Config config) throws Throwable {
        // Does nothing
    }

    public void clear() {
        prefixToContent.clear();
        prefixToUniqueId.clear();
        prefixToLastModified.clear();
    }

    @Override
    public void copyObject(DocumentStoreResource documentStoreResource, String destinationKey) {
        String contents = prefixToContent.get(documentStoreResource.getFullLocation());
        write(destinationKey, contents);
    }

    @Override
    public void deleteObjects(List<String> keys) {
        for (String key : keys) {
            if (prefixToContent.containsKey(key)) {
                prefixToContent.remove(key);
                prefixToUniqueId.remove(key);
                prefixToLastModified.remove(key);
            }
        }
    }

    private String getNextFolderAfterKey(String prefix, String objectKey) {
        int lengthOfBatchNumber = objectKey.substring(prefix.length()).indexOf("/");

        return objectKey.substring(0, prefix.length() + lengthOfBatchNumber + 1);
    }

    private long millisecondsSinceInitialization() {
        return Instant.now().toEpochMilli() - initializationTime.toEpochMilli();
    }
}
