package gov.irs.directfile.api.loaders.domain;

import java.util.List;
import java.util.Map;

public record TaxCompNode(String typeName, Map<String, String> options, List<TaxCompNode> children) {
    public TaxCompNode {
        options = Map.copyOf(options);
        children = List.copyOf(children);
    }
}
