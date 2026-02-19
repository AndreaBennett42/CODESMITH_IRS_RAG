package gov.irs.directfile.models;

import java.util.Date;
import java.util.UUID;

import lombok.Data;

@Data
public class TaxReturnSubmissionReceipt {
    private UUID taxReturnId;
    private String submissionId;
    private String receiptId;
    private Date submissionReceivedAt;

    public TaxReturnSubmissionReceipt(
            UUID taxReturnId, String submissionId, String receiptId, Date submissionReceivedAt) {
        this.taxReturnId = taxReturnId;
        this.submissionId = submissionId;
        this.receiptId = receiptId;
        this.submissionReceivedAt = submissionReceivedAt;
    }

    public TaxReturnSubmissionReceipt() {}
}
