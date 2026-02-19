package gov.irs.directfile.api.taxreturn.submissions.handlers.confirmation;

import java.util.ArrayList;
import java.util.List;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Slf4j
@Service
public class SubmissionConfirmationV2Handler implements SubmissionConfirmationHandler {
    private final ConfirmationService confirmationService;

    public SubmissionConfirmationV2Handler(ConfirmationService confirmationService) {
        this.confirmationService = confirmationService;
    }

    @Override
    public void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message) {
        List<TaxReturnSubmissionReceipt> submittedTaxReturnSubmissionReceipts = new ArrayList<>();
        List<SubmissionConfirmationPayloadV2Entry> failedTaxReturnSubmissionReceipts = new ArrayList<>();

        SubmissionConfirmationPayloadV2 payload = (SubmissionConfirmationPayloadV2) message.getPayload();
        payload.getEntries().forEach(entry -> {
            if (SubmissionEventTypeEnum.SUBMITTED.equals(entry.getEventType())) {
                submittedTaxReturnSubmissionReceipts.add(entry.getTaxReturnSubmissionReceipt());
            } else if (SubmissionEventTypeEnum.FAILED.equals(entry.getEventType())) {
                failedTaxReturnSubmissionReceipts.add(entry);
            }
        });

        if (!submittedTaxReturnSubmissionReceipts.isEmpty()) {
            log.info("Handling {} submission confirmations", submittedTaxReturnSubmissionReceipts.size());
            confirmationService.handleSubmissionConfirmations(submittedTaxReturnSubmissionReceipts);
        }

        if (!failedTaxReturnSubmissionReceipts.isEmpty()) {
            log.info("Handling {} submission failures", failedTaxReturnSubmissionReceipts.size());
            confirmationService.handleSubmissionFailures(failedTaxReturnSubmissionReceipts);
        }
    }
}
