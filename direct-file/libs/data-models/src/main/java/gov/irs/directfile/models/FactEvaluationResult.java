package gov.irs.directfile.models;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Pattern;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class FactEvaluationResult implements Serializable {
    // Regex to detect and parse collection indexing paths like `/claimedDependentsCollection/[1]/tin`
    private static final Pattern collectionIndexRegEx =
            Pattern.compile("(?<collection>.+)\\/\\[(?<index>\\d+)\\](?<suffix>\\/.+)");
    private Map<String, EvaluatedFactInfo> evaluatedFacts;
    private ObjectMapper objectMapper;

    public FactEvaluationResult() {
        this.evaluatedFacts = new HashMap<String, EvaluatedFactInfo>();
        this.objectMapper = new ObjectMapper();
    }

    public FactEvaluationResult(final Map<String, EvaluatedFactInfo> evaluatedFacts) {
        this.evaluatedFacts = new HashMap<>(evaluatedFacts);
        this.objectMapper = new ObjectMapper();
    }

    public Map<String, EvaluatedFactInfo> getEvaluatedFacts() {
        return evaluatedFacts;
    }

    public int size() {
        return this.evaluatedFacts.size();
    }

    public Iterator<Map.Entry<String, EvaluatedFactInfo>> iterator() {
        return this.evaluatedFacts.entrySet().iterator();
    }

    // Treats a missing/undefined boolean fact as `false`.
    public boolean getBoolean(final String factPath) {
        if (factPath == null) return false;
        final String _factPath = collectionIndexToId(factPath);
        final var optional = this.getOptional(_factPath);
        if (optional.isEmpty()) return false;
        final var object = optional.get();
        if (object.getClass() == String.class) return Boolean.parseBoolean((String) object);
        return (boolean) object;
    }

    public Optional<Object> getOptional(final String factPath) {
        final String _factPath = collectionIndexToId(factPath);
        final EvaluatedFactInfo efi = this.evaluatedFacts.get(_factPath);
        if (efi == null) return Optional.empty();
        final Object objectValue = efi.value();
        final String asString = objectValue == null ? null : objectValue.toString();
        // Remove leading, trailing, and repeating spaces.
        final String asCleanString = asString == null ? null : asString.trim().replaceAll("\\s+", " ");
        if (objectValue == null || asString == null || asCleanString.length() == 0) {
            return Optional.empty();
        }
        if (objectValue.getClass() == String.class) {
            return Optional.of(asCleanString);
        }
        return Optional.of(objectValue);
    }

    public String getString(final String factPath) {
        final String _factPath = collectionIndexToId(factPath);
        final EvaluatedFactInfo efi = this.evaluatedFacts.get(_factPath);
        if (efi == null) return "";

        final Object objectValue = efi.value();
        if (objectValue == null || objectValue.toString() == null) {
            return "";
        }

        // Remove leading, trailing, and repeating spaces.
        final String value = objectValue.toString().trim().replaceAll("\\s+", " ");

        return value;
    }

    public void put(final String path, final EvaluatedFactInfo value) {
        this.evaluatedFacts.put(collectionIndexToId(path), value);
    }

    public void put(final String path, final Object value) {
        final String type =
                (value != null && value.getClass() == Boolean.class) ? "java.lang.Boolean" : "java.lang.String";
        this.evaluatedFacts.put(collectionIndexToId(path), new EvaluatedFactInfo(type, value));
    }

    public String toJson() throws JsonProcessingException {
        return objectMapper.writeValueAsString(evaluatedFacts);
    }

    // Implements a syntax for collection indexing paths by transforming inputs like `/someCollection/[N]/foo` (where N
    // is a non-negative integer) into valid dictionary paths like `/someCollection/#ID/foo` (where ID is the UUID for
    // the (N-1)th collection item).
    // Inputs of valid dictionary paths are returned unchanged.
    private String collectionIndexToId(final String factPath) {
        if (factPath == null) return null;
        final var matcher = collectionIndexRegEx.matcher(factPath);
        if (!matcher.find()) return factPath;
        final String collectionPath = matcher.group("collection");
        final String indexString = matcher.group("index");
        final String pathSuffix = matcher.group("suffix");
        final Optional<Object> optional = this.getOptional(collectionPath);
        if (optional.isEmpty()) return factPath;

        int index = -1;
        try {
            index = Integer.parseInt(indexString);
        } catch (NumberFormatException e) {
            return factPath;
        }
        final var idList = (List<UUID>) optional.get();
        if (index < 0 || index >= idList.size()) return factPath;
        final var id = idList.get(index);
        return String.format("%s/#%s%s", collectionPath, id.toString(), pathSuffix);
    }

    // Converts collection indexing paths to wildcard path.
    // Example: `/collection/[0]/foo` becomes `/collection/*/foo`
    public static String collectionIndexToWildcard(final String factPath) {
        final var matcher = collectionIndexRegEx.matcher(factPath);
        if (!matcher.find()) return factPath;
        final String collectionPath = matcher.group("collection");
        final String pathSuffix = matcher.group("suffix");
        return String.format("%s/*%s", collectionPath, pathSuffix);
    }

    public static boolean isPseudoPath(final String factPath) {
        return factPath.contains("~");
    }
}
