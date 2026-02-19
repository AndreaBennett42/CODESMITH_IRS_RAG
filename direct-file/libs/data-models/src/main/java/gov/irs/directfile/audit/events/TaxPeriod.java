package gov.irs.directfile.audit.events;

import com.fasterxml.jackson.annotation.JsonValue;

public enum TaxPeriod {
    TY2023("2023"),
    TY2024("2024");

    private final String value;

    TaxPeriod(String value) {
        this.value = value;
    }

    @Override
    @JsonValue
    public String toString() {
        return value;
    }
}
