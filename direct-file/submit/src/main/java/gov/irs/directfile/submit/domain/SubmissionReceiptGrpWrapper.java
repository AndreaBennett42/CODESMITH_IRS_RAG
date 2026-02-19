package gov.irs.directfile.submit.domain;

import javax.xml.datatype.XMLGregorianCalendar;

import lombok.Getter;
import lombok.Setter;

import gov.irs.mef.SubmissionReceiptList;

@Getter
@Setter
public class SubmissionReceiptGrpWrapper {
    private SubmissionReceiptList.SubmissionReceiptGrp submissionReceiptGrp;
    private final String submissionId;
    private final String receiptId;
    private final XMLGregorianCalendar submissionReceivedTs;

    SubmissionReceiptGrpWrapper(SubmissionReceiptList.SubmissionReceiptGrp submissionReceiptGrp) {
        this.submissionId = submissionReceiptGrp.getSubmissionId();
        this.receiptId = submissionReceiptGrp.getReceiptId();
        this.submissionReceivedTs = submissionReceiptGrp.getSubmissionReceivedTs();
    }

    public SubmissionReceiptGrpWrapper(
            String submissionId, String receiptId, XMLGregorianCalendar submissionReceivedTs) {
        this.submissionId = submissionId;
        this.receiptId = receiptId;
        this.submissionReceivedTs = submissionReceivedTs;
    }
}
