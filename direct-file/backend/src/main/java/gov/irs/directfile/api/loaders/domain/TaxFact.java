package gov.irs.directfile.api.loaders.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;

public record TaxFact(
        String path,
        @JsonIgnore String name,
        @JsonIgnore String description,
        boolean exportZero,
        TaxWritable writable,
        TaxCompNode derived,
        TaxCompNode placeholder,
        ExportNode export) {}
