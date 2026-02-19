package gov.irs.directfile.submit.domain;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.xml.datatype.XMLGregorianCalendar;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Slf4j
public class SubmittedDataContainer {
    public SubmittedDataContainer(
            List<UserContextData> userContexts,
            SendSubmissionsResultWrapper sendSubmissionsResultWrapper,
            SubmissionBatch submissionBatch) {
        this.userContexts = userContexts;
        this.sendSubmissionsResultWrapper = sendSubmissionsResultWrapper;
        this.submissionBatch = submissionBatch;
    }

    public List<UserContextData> userContexts;

    public SubmissionBatch submissionBatch;

    @Getter
    private SendSubmissionsResultWrapper sendSubmissionsResultWrapper;

    public List<TaxReturnIdAndSubmissionId> getTaxReturnIdAndSubmissionIds() {
        return userContexts.stream()
                .map(userContextData -> new TaxReturnIdAndSubmissionId(
                        UUID.fromString(userContextData.getTaxReturnId()), userContextData.getSubmissionId()))
                .toList();
    }

    public List<SubmissionConfirmationPayloadV2Entry> getSuccessSubmissionConfirmationPayloadV2Entries() {
        HashMap<String, String> submissionIdToTaxReturnIdMap = new HashMap<>();
        userContexts.forEach(userContextData ->
                submissionIdToTaxReturnIdMap.put(userContextData.getSubmissionId(), userContextData.getTaxReturnId()));

        return sendSubmissionsResultWrapper.getReceipts().stream()
                .map(submissionReceiptGrpWrapper -> {
                    String submissionId = submissionReceiptGrpWrapper.getSubmissionId();
                    String taxReturnId = submissionIdToTaxReturnIdMap.get(submissionId);

                    if (taxReturnId == null) {
                        log.error(
                                "Unable to find taxReturnId by submissionId: {}. The submissionId sent to MeF may not match the submissionId in the MeF response",
                                submissionId);
                    }

                    String receiptId = submissionReceiptGrpWrapper.getReceiptId();
                    XMLGregorianCalendar submissionReceivedTs = submissionReceiptGrpWrapper.getSubmissionReceivedTs();

                    TaxReturnSubmissionReceipt taxReturnSubmissionReceipt = new TaxReturnSubmissionReceipt(
                            UUID.fromString(taxReturnId),
                            submissionId,
                            receiptId,
                            submissionReceivedTs.toGregorianCalendar().getTime());

                    return new SubmissionConfirmationPayloadV2Entry(
                            taxReturnSubmissionReceipt, SubmissionEventTypeEnum.SUBMITTED, Map.of());
                })
                .toList();
    }
}
