package gov.irs.directfile.status.services.handlers.confirmation;

import java.util.List;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV1;
import gov.irs.directfile.status.acknowledgement.PendingAcknowledgementRepository;
import gov.irs.directfile.status.acknowledgement.TaxReturnSubmissionRepository;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.Pending;
import gov.irs.directfile.status.domain.TaxReturnSubmission;

/**
 * @deprecated V1 messages are no longer sent, but keeping handlers until we are sure queues are V1-free
 */
@Deprecated
@Service
@Slf4j
public class SubmissionConfirmationV1Handler implements SubmissionConfirmationHandler {
    private final PendingAcknowledgementRepository pendingRepo;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepo;
    private final StatusProperties statusProperties;

    public SubmissionConfirmationV1Handler(
            PendingAcknowledgementRepository pendingRepo,
            TaxReturnSubmissionRepository taxReturnSubmissionRepo,
            StatusProperties statusProperties) {

        this.pendingRepo = pendingRepo;
        this.taxReturnSubmissionRepo = taxReturnSubmissionRepo;
        this.statusProperties = statusProperties;
    }

    @Override
    public void handleSubmissionConfirmationMessage(
            VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message) {
        SubmissionConfirmationPayloadV1 payload = (SubmissionConfirmationPayloadV1) message.getPayload();
        List<TaxReturnSubmissionReceipt> taxReturnSubmissionReceipts = payload.getReceipts();

        StringBuilder sb = new StringBuilder();
        sb.append("Received submission confirmation V1 message for tax return ids:");
        taxReturnSubmissionReceipts.forEach(
                taxReturnSubmissionReceipt -> sb.append(" ").append(taxReturnSubmissionReceipt.getTaxReturnId()));
        log.info(sb.toString());

        List<Pending> pendings = taxReturnSubmissionReceipts.stream()
                .map(taxReturnSubmissionReceipt ->
                        new Pending(taxReturnSubmissionReceipt.getSubmissionId(), statusProperties.getApplicationId()))
                .toList();

        List<TaxReturnSubmission> taxReturnSubmissions = taxReturnSubmissionReceipts.stream()
                .map(taxReturnSubmissionReceipt -> new TaxReturnSubmission(
                        taxReturnSubmissionReceipt.getTaxReturnId(), taxReturnSubmissionReceipt.getSubmissionId()))
                .toList();

        log.info(String.format("Saving %s pendings", pendings.size()));
        pendingRepo.saveAll(pendings);
        taxReturnSubmissionRepo.saveAll(taxReturnSubmissions);
    }
}
