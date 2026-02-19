package gov.irs.directfile.api.loaders.domain;

public record TaxLimit(String operation, TaxLimitLevel level, TaxCompNode node) {}
