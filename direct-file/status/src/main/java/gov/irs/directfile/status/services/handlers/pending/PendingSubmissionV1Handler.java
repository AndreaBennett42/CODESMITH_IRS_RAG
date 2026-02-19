package gov.irs.directfile.status.services.handlers.pending;

import java.util.List;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.models.message.pending.payload.PendingSubmissionPayloadV1;
import gov.irs.directfile.status.acknowledgement.PendingAcknowledgementRepository;
import gov.irs.directfile.status.acknowledgement.TaxReturnSubmissionRepository;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.Pending;
import gov.irs.directfile.status.domain.TaxReturnSubmission;

@Service
@Slf4j
public class PendingSubmissionV1Handler implements PendingSubmissionHandler {
    private final PendingAcknowledgementRepository pendingRepo;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepo;
    private final StatusProperties statusProperties;

    public PendingSubmissionV1Handler(
            PendingAcknowledgementRepository pendingRepo,
            TaxReturnSubmissionRepository taxReturnSubmissionRepo,
            StatusProperties statusProperties) {
        this.pendingRepo = pendingRepo;
        this.taxReturnSubmissionRepo = taxReturnSubmissionRepo;
        this.statusProperties = statusProperties;
    }

    @Override
    public void handlePendingSubmissionMessage(
            VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> message) {
        PendingSubmissionPayloadV1 payload = (PendingSubmissionPayloadV1) message.getPayload();
        List<TaxReturnIdAndSubmissionId> taxReturnIdAndSubmissionIds = payload.getPendings();

        StringBuilder sb = new StringBuilder();
        sb.append("Received pending submission V1 message for tax return ids:");
        taxReturnIdAndSubmissionIds.forEach(
                taxReturnIdAndSubmissionId -> sb.append(" ").append(taxReturnIdAndSubmissionId.getTaxReturnId()));
        log.info(sb.toString());

        List<Pending> pendings = taxReturnIdAndSubmissionIds.stream()
                .map(taxReturnIdAndSubmissionId -> {
                    Pending pending = new Pending();
                    pending.setSubmissionId(taxReturnIdAndSubmissionId.getSubmissionId());
                    pending.setPodId(statusProperties.getApplicationId());
                    return pending;
                })
                .toList();

        List<TaxReturnSubmission> taxReturnSubmissions = taxReturnIdAndSubmissionIds.stream()
                .map(taxReturnIdAndSubmissionId -> new TaxReturnSubmission(
                        taxReturnIdAndSubmissionId.getTaxReturnId(), taxReturnIdAndSubmissionId.getSubmissionId()))
                .toList();

        log.info(String.format("Saving %s pendings", pendings.size()));
        pendingRepo.saveAll(pendings);
        taxReturnSubmissionRepo.saveAll(taxReturnSubmissions);
    }
}
