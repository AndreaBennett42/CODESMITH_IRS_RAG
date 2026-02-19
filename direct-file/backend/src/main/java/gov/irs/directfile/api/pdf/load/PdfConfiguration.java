package gov.irs.directfile.api.pdf.load;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.directfile.api.pdf.PdfTableFactExpressionMapping;
import gov.irs.directfile.api.pdf.TableConfig;

@Getter
@AllArgsConstructor
@SuppressFBWarnings(
        value = {"MS_SHOULD_BE_FINAL"},
        justification = "Initial Spotbugs setup")
@SuppressWarnings(value = {"PMD.AssignmentInOperand", "PMD.AvoidReassigningLoopVariables"})
public class PdfConfiguration {
    private String includeWhenFactPath;
    private String includeForEachCollectionFactPath;
    private Map<String, String> mapPdfFieldsToFactExpressions;
    private Set<String> factPathsForPdf;
    private TableConfig tableConfig;
    private Map<String, Map<String, String>> customData;

    public static PdfConfiguration load(InputStream configStream) throws IOException {
        final ByteArrayOutputStream result = new ByteArrayOutputStream();
        // just taking a meg on load.
        // I won't keep it.
        final byte[] buffer = new byte[1024];
        for (int length; (length = configStream.read(buffer)) != -1; ) {
            result.write(buffer, 0, length);
        }

        final String all = result.toString(StandardCharsets.UTF_8.name());
        final ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        final JsonNode nodes = mapper.readTree(all);
        Map<String, String> mapFieldsToExpressions = new HashMap<String, String>();
        Set<String> factPaths = new HashSet<String>();

        // The config should have a top-level key for either ...
        // ... `includeWhen`, a fact that, when true, means include the form, or ...
        final JsonNode includeWhen = nodes.get("includeWhen");
        // ... `includeForEach`, a collection; include a copy of form for each item.
        final JsonNode includeForEach = nodes.get("includeForEach");

        if (includeWhen != null) factPaths.add(includeWhen.asText());
        if (includeForEach != null) factPaths.add(includeForEach.asText());

        // An optional top level key is for an array of facts that are not displayed,
        // but used to compute pseudofacts.
        final JsonNode otherRequiredFacts = nodes.get("otherRequiredFacts");
        if (otherRequiredFacts != null) {
            final Iterator<JsonNode> factsIter = otherRequiredFacts.elements();
            while (factsIter.hasNext()) {
                factPaths.add(factsIter.next().asText());
            }
        }

        // Another optional top level key `customMaps` allows for arbitrary data.
        final JsonNode customMaps = nodes.get("customMaps");
        final Map<String, Map<String, String>> mapOfMaps = new HashMap<>();
        if (customMaps != null) {
            final var customMapsIter = customMaps.fields();
            while (customMapsIter.hasNext()) {
                final var _entry = customMapsIter.next();
                final String mapName = _entry.getKey();
                final Map<String, String> map = new LinkedHashMap<>();
                final var customMapIter = _entry.getValue().fields();
                while (customMapIter.hasNext()) {
                    final var entry = customMapIter.next();
                    map.put(entry.getKey(), entry.getValue().asText());
                }
                mapOfMaps.put(mapName, map);
            }
        }

        // Another top level key indicates the format, ...
        // ... either a `table` with columns for each item in a collection, ...
        final JsonNode table = nodes.get("table");
        if (table != null) {
            final JsonNode rowsCollectionPath = table.get("rowsCollectionPath");
            factPaths.add(rowsCollectionPath.asText());
            final JsonNode itemsToSkip = table.get("itemsToSkip");
            final JsonNode rowsPerPage = table.get("rowsPerPage");
            final JsonNode columns = table.get("columns");
            final List<PdfTableFactExpressionMapping> columnList = new ArrayList<PdfTableFactExpressionMapping>();
            final Iterator<JsonNode> columnIter = columns.elements();
            while (columnIter.hasNext()) {
                final JsonNode column = columnIter.next();
                final String factExpression = column.get("factExpression")
                        .asText()
                        .replaceAll(Pattern.quote("../"), rowsCollectionPath.asText() + "/*/");
                Pattern factPathRegEx = Pattern.compile("[/~]\\S+");
                Matcher matcher = factPathRegEx.matcher(factExpression);
                for (int i = 0; matcher.find(i); i = matcher.end()) {
                    factPaths.add(matcher.group());
                }
                columnList.add(new PdfTableFactExpressionMapping(
                        factExpression, column.get("fieldName").asText()));
            }
            // Optionally, tables can specify fact expressions to appear once per page
            // (header, footer, etc.)
            final JsonNode oncePerPage = table.get("oncePerPage");
            final List<PdfTableFactExpressionMapping> oncePerPageList = new ArrayList<PdfTableFactExpressionMapping>();
            if (oncePerPage != null) {
                final Iterator<JsonNode> oncePerPageIter = oncePerPage.elements();
                while (oncePerPageIter.hasNext()) {
                    final JsonNode oncePerPageField = oncePerPageIter.next();
                    String factPath = oncePerPageField.get("factExpression").asText();
                    oncePerPageList.add(new PdfTableFactExpressionMapping(
                            factPath, oncePerPageField.get("fieldName").asText()));
                    factPaths.add(factPath);
                }
            }
            return new PdfConfiguration(
                    includeWhen == null ? null : includeWhen.asText(),
                    includeForEach == null ? null : includeForEach.asText(),
                    mapFieldsToExpressions,
                    factPaths,
                    new TableConfig(
                            rowsCollectionPath.asText(),
                            itemsToSkip == null ? 0 : itemsToSkip.asInt(0),
                            rowsPerPage.asInt(),
                            columnList,
                            oncePerPageList),
                    mapOfMaps);
        }

        // ... or a `form`, with a map of facts to fields.
        final JsonNode form = nodes.get("form");
        if (form != null) mapFieldsToExpressions = recursePdfField("", form, new HashMap<>());

        // Build a set of fact paths to extract for this type of PDF form.
        Pattern factPathRegEx = Pattern.compile("[/~]\\S+");
        for (String factExpression : mapFieldsToExpressions.values()) {
            Matcher matcher = factPathRegEx.matcher(factExpression);
            for (int i = 0; matcher.find(i); i = matcher.end()) {
                factPaths.add(matcher.group());
            }
        }
        return new PdfConfiguration(
                includeWhen == null ? null : includeWhen.asText(),
                includeForEach == null ? null : includeForEach.asText(),
                mapFieldsToExpressions,
                factPaths,
                null,
                mapOfMaps);
    }

    static Map<String, String> recursePdfField(String currentPath, JsonNode node, Map<String, String> output) {
        var fields = node.fields();
        while (fields.hasNext()) {
            var n = fields.next();
            if (n.getValue().isObject()) {
                recursePdfField(
                        currentPath.isBlank() ? n.getKey() : String.format("%s.%s", currentPath, n.getKey()),
                        n.getValue(),
                        output);
            } else {
                String valueText = n.getValue().asText();
                // Yes, that's a string comparison for the 4 character string "null"
                if (!"null".equals(valueText)) {
                    output.put((currentPath.isBlank() ? "" : currentPath + ".") + n.getKey(), valueText);
                }
            }
        }
        return output;
    }
}
