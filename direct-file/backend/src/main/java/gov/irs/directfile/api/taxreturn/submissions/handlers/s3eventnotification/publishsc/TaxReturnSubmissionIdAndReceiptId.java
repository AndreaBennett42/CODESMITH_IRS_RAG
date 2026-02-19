package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class TaxReturnSubmissionIdAndReceiptId {
    private String submissionId;
    private String receiptId;

    @Override
    public String toString() {
        return "{" + "submissionId='" + submissionId + '\'' + ", receiptId='" + receiptId + '\'' + '}';
    }
}
