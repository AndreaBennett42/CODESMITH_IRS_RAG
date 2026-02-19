package gov.irs.directfile.api.stateapi.domain.export;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import gov.irs.directfile.models.EvaluatedFactInfo;

public class ExportableCollection extends HashMap<String, ExportableCollectionItem>
        implements ExportableData<Map<String, ExportableCollectionItem>> {

    private static final String FILERS = "filers";
    private static final String MARRIED_FILING_JOINTLY_OPTION = "marriedFilingJointly";
    private static final String MARRIED_FILING_SEPARATELY_OPTION = "marriedFilingSeparately";
    private static final List<String> FILING_STATUS_OPTIONS_TO_INCLUDE_SECONDARY_FILER =
            List.of(MARRIED_FILING_JOINTLY_OPTION, MARRIED_FILING_SEPARATELY_OPTION);
    private static final String IS_PRIMARY_FILER = "isPrimaryFiler";

    public static class SanitizedExportableCollection extends ArrayList<ExportableCollectionItem>
            implements ExportableData<List<ExportableCollectionItem>> {

        @Override
        public List<ExportableCollectionItem> getValue() {
            return this;
        }
    }

    @Override
    public HashMap<String, ExportableCollectionItem> getValue() {
        return this;
    }

    public void putCollectionFact(String collectionUuid, String factKey, EvaluatedFactInfo evaluatedFactInfo) {
        var collectionItemToUpdate = this.get(collectionUuid);
        if (collectionItemToUpdate == null) {
            // This is the first fact for this collectionUuid, add a new collection item
            var collectionItem = new ExportableCollectionItem();
            collectionItem.put(factKey, evaluatedFactInfo);
            this.put(collectionUuid, collectionItem);
        } else {
            // The fact belongs to an existing collection item, add it
            collectionItemToUpdate.put(factKey, evaluatedFactInfo);
        }
    }

    public SanitizedExportableCollection sanitized(String collectionKey, String filingStatus) {
        // transform from map to list of items
        var sanitizedValuesStream = this.values().stream();

        // filter out placeholder filer for non-MFJ returns
        if (FILERS.equals(collectionKey) && shouldFilterOutPlaceholderFiler(filingStatus)) {
            sanitizedValuesStream = sanitizedValuesStream.filter(collectionItem -> collectionItem.containsKey(
                            IS_PRIMARY_FILER)
                    && Boolean.TRUE.equals(collectionItem.get(IS_PRIMARY_FILER).getValue()));
        }

        return sanitizedValuesStream
                .map(ExportableCollectionItem::sanitized)
                // filter out empty collectionItems
                .filter((collectionItem) -> !collectionItem.isEmpty())
                .collect(Collectors.toCollection(SanitizedExportableCollection::new));
    }

    private boolean shouldFilterOutPlaceholderFiler(String filingStatus) {
        return !FILING_STATUS_OPTIONS_TO_INCLUDE_SECONDARY_FILER.contains(filingStatus);
    }
}
