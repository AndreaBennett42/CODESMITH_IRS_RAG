package gov.irs.directfile.models.message;

import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;
import lombok.NonNull;

@Getter
@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum SubmissionEventFailureDetailEnum implements SubmissionEventFailureInterface {
    XML_VALIDATION("xml_validation"),
    SERVICE_UNAVAILABLE("service_unavailable"),
    SUBMISSION_PROCESSING("submission_processing");

    @JsonValue
    final String failureDetail;

    SubmissionEventFailureDetailEnum(String failureDetail) {
        this.failureDetail = failureDetail;
    }

    public static SubmissionEventFailureDetailEnum getEnum(@NonNull String value) {
        for (SubmissionEventFailureDetailEnum failureDetail : SubmissionEventFailureDetailEnum.values()) {
            if (failureDetail.name().equalsIgnoreCase(value)) return failureDetail;
        }
        throw new IllegalArgumentException("no valid enum constant for the specified value");
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
