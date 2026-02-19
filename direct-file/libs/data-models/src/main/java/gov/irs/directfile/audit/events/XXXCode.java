package gov.irs.directfile.audit.events;

import com.fasterxml.jackson.annotation.JsonValue;

public enum XXXCode {
    FORM_1040("XX");

    private final String value;

    XXXCode(String value) {
        this.value = value;
    }

    @Override
    @JsonValue
    public String toString() {
        return value;
    }
}
