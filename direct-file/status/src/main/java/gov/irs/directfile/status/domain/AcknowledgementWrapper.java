package gov.irs.directfile.status.domain;

import java.util.ArrayList;
import java.util.List;

import lombok.Getter;

import gov.irs.mef.AcknowledgementList;

@Getter
public class AcknowledgementWrapper {
    private final String submissionId;
    private final String receiptId;
    private final String acceptanceStatusTxt;
    private final List<ValidationErrorGrpWrapper> validationErrorList;

    public AcknowledgementWrapper(String submissionId, String receiptId, String acceptanceStatusTxt) {
        this(submissionId, receiptId, acceptanceStatusTxt, List.of());
    }

    public AcknowledgementWrapper(
            String submissionId,
            String receiptId,
            String acceptanceStatusTxt,
            List<ValidationErrorGrpWrapper> validationErrorList) {
        this.submissionId = submissionId;
        this.receiptId = receiptId;
        this.acceptanceStatusTxt = acceptanceStatusTxt;
        this.validationErrorList = List.copyOf(validationErrorList);
    }

    public static AcknowledgementWrapper fromMefAcknowledgement(AcknowledgementList.Acknowledgement acknowledgement) {
        if (acknowledgement.getValidationErrorList() == null) {
            return new AcknowledgementWrapper(
                    acknowledgement.getSubmissionId(),
                    acknowledgement.getReceiptId(),
                    acknowledgement.getAcceptanceStatusTxt(),
                    new ArrayList<>());
        } else {
            List<ValidationErrorGrpWrapper> validationErrors =
                    acknowledgement.getValidationErrorList().getValidationErrorGrp().stream()
                            .map(ValidationErrorGrpWrapper::fromMefValidationErrorGrp)
                            .toList();
            return new AcknowledgementWrapper(
                    acknowledgement.getSubmissionId(),
                    acknowledgement.getReceiptId(),
                    acknowledgement.getAcceptanceStatusTxt(),
                    validationErrors);
        }
    }
}
