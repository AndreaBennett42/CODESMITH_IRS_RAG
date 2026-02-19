package gov.irs.directfile.api.loaders.domain;

import java.util.Map;

import gov.irs.directfile.models.FactTypeWithItem;

public record ValidateFactsRequest(Map<String, FactTypeWithItem> facts) {
    public ValidateFactsRequest {
        facts = Map.copyOf(facts);
    }
}
