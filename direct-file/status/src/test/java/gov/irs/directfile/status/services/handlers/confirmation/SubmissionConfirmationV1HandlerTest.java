package gov.irs.directfile.status.services.handlers.confirmation;

import java.util.Date;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.SubmissionConfirmationMessageVersion;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV1;
import gov.irs.directfile.status.acknowledgement.PendingAcknowledgementRepository;
import gov.irs.directfile.status.acknowledgement.TaxReturnSubmissionRepository;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.Pending;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@EnableConfigurationProperties(StatusProperties.class)
@DataJpaTest
class SubmissionConfirmationV1HandlerTest {
    @Autowired
    PendingAcknowledgementRepository pendingRepo;

    @Autowired
    TaxReturnSubmissionRepository taxReturnSubmissionRepo;

    @Autowired
    private StatusProperties statusProperties;

    @Test
    public void handleSubmissionConfirmationMessage() {
        SubmissionConfirmationV1Handler handler =
                new SubmissionConfirmationV1Handler(pendingRepo, taxReturnSubmissionRepo, statusProperties);

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(UUID.randomUUID(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(UUID.randomUUID(), "submissionId2", "receiptId2", new Date());
        List<TaxReturnSubmissionReceipt> taxReturnSubmissionReceipts =
                List.of(taxReturnSubmissionReceipt1, taxReturnSubmissionReceipt2);

        AbstractSubmissionConfirmationPayload payload =
                new SubmissionConfirmationPayloadV1(taxReturnSubmissionReceipts);
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> queueMessage =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V1.getVersion()));

        handler.handleSubmissionConfirmationMessage(queueMessage);

        assertEquals(2, pendingRepo.count());
        Optional<Pending> pending1 = pendingRepo.GetPendingSubmission(taxReturnSubmissionReceipt1.getSubmissionId());
        assertTrue(pending1.isPresent()
                && pending1.get().getSubmissionId().equals(taxReturnSubmissionReceipt1.getSubmissionId()));
        Optional<Pending> pending2 = pendingRepo.GetPendingSubmission(taxReturnSubmissionReceipt2.getSubmissionId());
        assertTrue(pending2.isPresent()
                && pending2.get().getSubmissionId().equals(taxReturnSubmissionReceipt2.getSubmissionId()));

        assertEquals(2, taxReturnSubmissionRepo.count());
        Optional<String> submissionId1 = taxReturnSubmissionRepo.getLatestSubmissionIdByTaxReturnId(
                taxReturnSubmissionReceipt1.getTaxReturnId());
        assertTrue(
                submissionId1.isPresent() && submissionId1.get().equals(taxReturnSubmissionReceipt1.getSubmissionId()));
        Optional<String> submissionId2 = taxReturnSubmissionRepo.getLatestSubmissionIdByTaxReturnId(
                taxReturnSubmissionReceipt2.getTaxReturnId());
        assertTrue(
                submissionId2.isPresent() && submissionId2.get().equals(taxReturnSubmissionReceipt2.getSubmissionId()));
    }
}
