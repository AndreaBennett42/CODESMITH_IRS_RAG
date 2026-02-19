package gov.irs.directfile.status.services.handlers.confirmation;

import java.util.ArrayList;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;
import gov.irs.directfile.status.acknowledgement.PendingAcknowledgementRepository;
import gov.irs.directfile.status.acknowledgement.TaxReturnSubmissionRepository;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.Pending;
import gov.irs.directfile.status.domain.TaxReturnSubmission;

@Service
@Slf4j
@AllArgsConstructor
public class SubmissionConfirmationV2Handler implements SubmissionConfirmationHandler {
    private final PendingAcknowledgementRepository pendingRepo;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepo;
    private final StatusProperties statusProperties;

    @Override
    public void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message) {
        SubmissionConfirmationPayloadV2 payload = (SubmissionConfirmationPayloadV2) message.getPayload();

        // Note that for the pending submission queue, we only need to process SUBMITTED tax return
        // events.  FAILED events are also sent via the submission confirmation SNS topic subscription,
        // but can be ignored here since they aren't considered pending submissions.
        List<TaxReturnSubmissionReceipt> submittedTaxReturnSubmissionReceipts = new ArrayList<>();
        payload.getEntries().forEach(entry -> {
            if (SubmissionEventTypeEnum.SUBMITTED.equals(entry.getEventType())) {
                submittedTaxReturnSubmissionReceipts.add(entry.getTaxReturnSubmissionReceipt());
            }
        });

        // If we did not get any SUBMITTED returns, then we're done.
        if (submittedTaxReturnSubmissionReceipts.isEmpty()) return;

        // Otherwise, save the database records for the SUBMITTED returns.
        StringBuilder sb = new StringBuilder();
        sb.append("Received SUBMITTED submission confirmation V2 message for tax return ids:");
        submittedTaxReturnSubmissionReceipts.forEach(
                taxReturnSubmissionReceipt -> sb.append(" ").append(taxReturnSubmissionReceipt.getTaxReturnId()));
        log.info(sb.toString());

        List<Pending> pendings = submittedTaxReturnSubmissionReceipts.stream()
                .map(taxReturnSubmissionReceipt ->
                        new Pending(taxReturnSubmissionReceipt.getSubmissionId(), statusProperties.getApplicationId()))
                .toList();

        List<TaxReturnSubmission> taxReturnSubmissions = submittedTaxReturnSubmissionReceipts.stream()
                .map(taxReturnSubmissionReceipt -> new TaxReturnSubmission(
                        taxReturnSubmissionReceipt.getTaxReturnId(), taxReturnSubmissionReceipt.getSubmissionId()))
                .toList();

        log.info(String.format("Saving %s pendings", pendings.size()));
        pendingRepo.saveAll(pendings);
        taxReturnSubmissionRepo.saveAll(taxReturnSubmissions);
    }
}
