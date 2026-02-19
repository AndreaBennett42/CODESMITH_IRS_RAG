package gov.irs.directfile.api.loaders.domain;

import java.util.List;
import java.util.Map;

public record TaxWritable(
        String typeName, Map<String, String> options, String collectionItemAlias, List<TaxLimit> limits) {
    public TaxWritable {
        options = Map.copyOf(options);
        limits = List.copyOf(limits);
    }
}
