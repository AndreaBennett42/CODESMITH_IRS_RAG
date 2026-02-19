package gov.irs.directfile.models;

import java.util.UUID;

import lombok.Data;

@Data
public class TaxReturnIdAndSubmissionId {
    private UUID taxReturnId;
    private String submissionId;

    public TaxReturnIdAndSubmissionId(UUID taxReturnId, String submissionId) {
        this.taxReturnId = taxReturnId;
        this.submissionId = submissionId;
    }

    public TaxReturnIdAndSubmissionId() {}
}
