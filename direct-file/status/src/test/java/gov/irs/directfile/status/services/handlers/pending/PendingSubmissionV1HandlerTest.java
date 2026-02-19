package gov.irs.directfile.status.services.handlers.pending;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.pending.PendingSubmissionMessageVersion;
import gov.irs.directfile.models.message.pending.VersionedPendingSubmissionMessage;
import gov.irs.directfile.models.message.pending.payload.AbstractPendingSubmissionPayload;
import gov.irs.directfile.models.message.pending.payload.PendingSubmissionPayloadV1;
import gov.irs.directfile.status.acknowledgement.PendingAcknowledgementRepository;
import gov.irs.directfile.status.acknowledgement.TaxReturnSubmissionRepository;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.Pending;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@EnableConfigurationProperties(StatusProperties.class)
@DataJpaTest
class PendingSubmissionV1HandlerTest {
    @Autowired
    PendingAcknowledgementRepository pendingRepo;

    @Autowired
    TaxReturnSubmissionRepository taxReturnSubmissionRepo;

    @Autowired
    StatusProperties statusProperties;

    @Test
    public void handlePendingSubmissionMessage() {
        PendingSubmissionV1Handler handler =
                new PendingSubmissionV1Handler(pendingRepo, taxReturnSubmissionRepo, statusProperties);

        TaxReturnIdAndSubmissionId taxReturnIdAndSubmissionId1 =
                new TaxReturnIdAndSubmissionId(UUID.randomUUID(), "submissionId1");
        TaxReturnIdAndSubmissionId taxReturnIdAndSubmissionId2 =
                new TaxReturnIdAndSubmissionId(UUID.randomUUID(), "submissionId2");
        List<TaxReturnIdAndSubmissionId> taxReturnIdAndSubmissionIds =
                List.of(taxReturnIdAndSubmissionId1, taxReturnIdAndSubmissionId2);

        AbstractPendingSubmissionPayload payload = new PendingSubmissionPayloadV1(taxReturnIdAndSubmissionIds);
        VersionedPendingSubmissionMessage<AbstractPendingSubmissionPayload> queueMessage =
                new VersionedPendingSubmissionMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        PendingSubmissionMessageVersion.V1.getVersion()));

        handler.handlePendingSubmissionMessage(queueMessage);

        assertEquals(2, pendingRepo.count());
        Optional<Pending> pending1 = pendingRepo.GetPendingSubmission(taxReturnIdAndSubmissionId1.getSubmissionId());
        assertTrue(pending1.isPresent()
                && pending1.get().getSubmissionId().equals(taxReturnIdAndSubmissionId1.getSubmissionId()));
        Optional<Pending> pending2 = pendingRepo.GetPendingSubmission(taxReturnIdAndSubmissionId2.getSubmissionId());
        assertTrue(pending2.isPresent()
                && pending2.get().getSubmissionId().equals(taxReturnIdAndSubmissionId2.getSubmissionId()));

        assertEquals(2, taxReturnSubmissionRepo.count());
        Optional<String> submissionId1 = taxReturnSubmissionRepo.getLatestSubmissionIdByTaxReturnId(
                taxReturnIdAndSubmissionId1.getTaxReturnId());
        assertTrue(
                submissionId1.isPresent() && submissionId1.get().equals(taxReturnIdAndSubmissionId1.getSubmissionId()));
        Optional<String> submissionId2 = taxReturnSubmissionRepo.getLatestSubmissionIdByTaxReturnId(
                taxReturnIdAndSubmissionId2.getTaxReturnId());
        assertTrue(
                submissionId2.isPresent() && submissionId2.get().equals(taxReturnIdAndSubmissionId2.getSubmissionId()));
    }
}
