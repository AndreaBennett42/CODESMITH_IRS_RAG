package gov.irs.directfile.api.loaders.domain;

import com.fasterxml.jackson.annotation.JsonValue;

@SuppressWarnings("PMD.AvoidThrowingNullPointerException")
public enum TaxLimitLevel {
    Warn("warn"),
    Error("error");

    public static final TaxLimitLevel DEFAULT_LEVEL = Error;

    public static TaxLimitLevel from(final String name) {
        if (name == null) {
            return DEFAULT_LEVEL;
        }
        for (TaxLimitLevel limitLevel : values()) {
            if (limitLevel.name().equalsIgnoreCase(name)) {
                return limitLevel;
            }
        }
        throw new NullPointerException(String.format("%s is not a valid TaxLimitLevel", name));
    }

    public final String stringValue;

    TaxLimitLevel(final String stringValue) {
        this.stringValue = stringValue;
    }

    @Override
    @JsonValue
    public String toString() {
        return stringValue;
    }
}
