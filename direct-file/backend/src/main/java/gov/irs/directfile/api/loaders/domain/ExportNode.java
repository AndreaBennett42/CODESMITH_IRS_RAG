package gov.irs.directfile.api.loaders.domain;

import java.util.Map;

public record ExportNode(String typeName, ExportNodeOptions options) {
    public ExportNode(String typeName, Map<String, String> optionsMap) {
        this(typeName, ExportNodeOptions.getExportNodeOptions(optionsMap));
    }
}
