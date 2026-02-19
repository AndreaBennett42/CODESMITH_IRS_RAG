package gov.irs.directfile.api.taxreturn;

import com.fasterxml.jackson.annotation.*;

@SuppressWarnings("PMD.AvoidDuplicateLiterals")
public enum ApiErrorKeys {
    UNEDITABLE_TAX_RETURN("uneditableTaxReturn"),
    // Update once custom message for errors related to the factgraph failing is ready
    SUBMISSION_BLOCKING_FACTS("generic.submissionError"),
    SUBMISSION_ERROR("generic.submissionError"),
    FACT_GRAPH_PARSE("generic.submissionError"),
    S3_WRITE_ERROR("externalServiceError");

    private final String value;

    ApiErrorKeys(String value) {
        this.value = value;
    }

    @Override
    @JsonValue
    public String toString() {
        return value;
    }
}
