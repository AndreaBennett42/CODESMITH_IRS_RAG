package gov.irs.directfile.api.loaders.domain;

import java.util.Map;

public record ExportNodeOptions(boolean downstreamFacts, boolean mef, boolean stateSystems) {

    static final String FACT_EXPORT_CHILD_DOWNSTREAM_FACT = "downstreamFacts";
    static final String FACT_EXPORT_CHILD_MEF_FACT = "mef";
    static final String FACT_EXPORT_CHILD_STATE_SYSTEMS_FACT = "stateSystems";

    public static ExportNodeOptions getExportNodeOptions(Map<String, String> optionsMap) {
        boolean downstreamFacts =
                Boolean.TRUE.toString().equalsIgnoreCase(optionsMap.get(FACT_EXPORT_CHILD_DOWNSTREAM_FACT));
        boolean mef = Boolean.TRUE.toString().equalsIgnoreCase(optionsMap.get(FACT_EXPORT_CHILD_MEF_FACT));
        boolean stateSystems =
                Boolean.TRUE.toString().equalsIgnoreCase(optionsMap.get(FACT_EXPORT_CHILD_STATE_SYSTEMS_FACT));

        return new ExportNodeOptions(downstreamFacts, mef, stateSystems);
    }
}
