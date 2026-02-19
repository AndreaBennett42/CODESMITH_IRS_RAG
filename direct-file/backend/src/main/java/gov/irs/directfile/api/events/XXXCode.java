package gov.irs.directfile.api.events;

import com.fasterxml.jackson.annotation.JsonValue;

public enum XXXCode {
    XXX_CODE("XXX");

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
