package gov.irs.directfile.models.message;

import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;
import lombok.NonNull;

@Getter
@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum SubmissionEventFailureCategoryEnum implements SubmissionEventFailureInterface {
    VALIDATION("validation"),
    PROCESSING("processing");

    @JsonValue
    final String failureCategory;

    SubmissionEventFailureCategoryEnum(String failureCategory) {
        this.failureCategory = failureCategory;
    }

    public static SubmissionEventFailureCategoryEnum getEnum(@NonNull String value) {
        for (SubmissionEventFailureCategoryEnum failureCategory : SubmissionEventFailureCategoryEnum.values()) {
            if (failureCategory.name().equalsIgnoreCase(value)) return failureCategory;
        }
        throw new IllegalArgumentException("no valid enum constant for the specified value");
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
