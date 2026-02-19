package gov.irs.directfile.api.loaders.domain;

import java.util.*;

import lombok.Getter;

@Getter
public class TaxDictionaryDigest {

    private final String sourceName;
    private final Map<String, TaxFact> facts;
    private final Set<String> exportZeroFacts;
    private final List<TaxFact> submissionBlockingFacts;
    private final List<TaxFact> exportToStatesFacts;

    public TaxDictionaryDigest(String sourceName, Map<String, TaxFact> facts) {
        this.sourceName = sourceName;
        this.facts = Map.copyOf(facts);

        exportZeroFacts = new HashSet<>();
        submissionBlockingFacts = new ArrayList<>();
        exportToStatesFacts = new ArrayList<>();

        this.facts.values().forEach(taxFact -> {
            if (taxFact.exportZero()) {
                exportZeroFacts.add(taxFact.path());
            }
            if (taxFact.export() != null
                    && taxFact.export().options() != null
                    && taxFact.export().options().stateSystems()) {
                exportToStatesFacts.add(taxFact);
            }
        });
    }
}
