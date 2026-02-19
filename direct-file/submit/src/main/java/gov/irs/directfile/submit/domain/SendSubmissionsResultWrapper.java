package gov.irs.directfile.submit.domain;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import lombok.Getter;
import lombok.Setter;

import gov.irs.mef.services.transmitter.mtom.SendSubmissionsResult;

@Getter
@Setter
public class SendSubmissionsResultWrapper {
    private SendSubmissionsResult submissionsResult;
    private List<SubmissionReceiptGrpWrapper> submissionReceiptGrpWrappers;

    public SendSubmissionsResultWrapper(SendSubmissionsResult submissionsResult) {
        this.submissionsResult = submissionsResult;

        if (sendSubmissionsResultHasReceipts(submissionsResult)) {
            this.submissionReceiptGrpWrappers = submissionsResult.getSubmissionReceiptList().getReceipts().stream()
                    .map(SubmissionReceiptGrpWrapper::new)
                    .collect(Collectors.toList());
        } else {
            this.submissionReceiptGrpWrappers = Collections.emptyList();
        }
    }

    public SendSubmissionsResultWrapper(List<SubmissionReceiptGrpWrapper> submissionReceiptGrpWrappers) {
        this.submissionReceiptGrpWrappers = submissionReceiptGrpWrappers;
    }

    private static boolean sendSubmissionsResultHasReceipts(SendSubmissionsResult submissionsResult) {
        return submissionsResult != null
                && submissionsResult.getSubmissionReceiptList() != null
                && submissionsResult.getSubmissionReceiptList().getReceipts() != null;
    }

    public List<SubmissionReceiptGrpWrapper> getReceipts() {
        return submissionReceiptGrpWrappers;
    }
}
