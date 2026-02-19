package gov.irs.directfile.audit.events;

import com.fasterxml.jackson.annotation.JsonValue;

public enum TinType {
    INDIVIDUAL("0");

    private final String value;

    TinType(String value) {
        this.value = value;
    }

    @Override
    @JsonValue
    public String toString() {
        return value;
    }
}
