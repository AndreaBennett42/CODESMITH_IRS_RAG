package gov.irs.directfile.status.services.handlers.confirmation;

import java.util.*;

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
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;
import gov.irs.directfile.status.acknowledgement.PendingAcknowledgementRepository;
import gov.irs.directfile.status.acknowledgement.TaxReturnSubmissionRepository;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.Pending;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@EnableConfigurationProperties(StatusProperties.class)
@DataJpaTest
class SubmissionConfirmationV2HandlerTest {
    @Autowired
    PendingAcknowledgementRepository pendingRepo;

    @Autowired
    TaxReturnSubmissionRepository taxReturnSubmissionRepo;

    @Autowired
    StatusProperties statusProperties;

    @Test
    public void handleSubmissionConfirmationMessage() {
        SubmissionConfirmationV2Handler handler =
                new SubmissionConfirmationV2Handler(pendingRepo, taxReturnSubmissionRepo, statusProperties);

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(UUID.randomUUID(), "submissionId1", "receiptId1", new Date());
        SubmissionConfirmationPayloadV2Entry entry1 = new SubmissionConfirmationPayloadV2Entry(
                taxReturnSubmissionReceipt1, SubmissionEventTypeEnum.SUBMITTED, Map.of());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(UUID.randomUUID(), "submissionId2", "receiptId2", new Date());
        SubmissionConfirmationPayloadV2Entry entry2 = new SubmissionConfirmationPayloadV2Entry(
                taxReturnSubmissionReceipt2, SubmissionEventTypeEnum.FAILED, Map.of());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt3 =
                new TaxReturnSubmissionReceipt(UUID.randomUUID(), "submissionId3", "receiptId3", new Date());
        SubmissionConfirmationPayloadV2Entry entry3 = new SubmissionConfirmationPayloadV2Entry(
                taxReturnSubmissionReceipt3, SubmissionEventTypeEnum.SUBMITTED, Map.of());

        List<SubmissionConfirmationPayloadV2Entry> entries = List.of(entry1, entry2, entry3);

        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV2(entries);
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> queueMessage =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));

        handler.handleSubmissionConfirmationMessage(queueMessage);

        assertEquals(2, pendingRepo.count());
        Optional<Pending> pending1 = pendingRepo.GetPendingSubmission(taxReturnSubmissionReceipt1.getSubmissionId());
        assertTrue(pending1.isPresent()
                && pending1.get().getSubmissionId().equals(taxReturnSubmissionReceipt1.getSubmissionId()));
        Optional<Pending> pending2 = pendingRepo.GetPendingSubmission(taxReturnSubmissionReceipt3.getSubmissionId());
        assertTrue(pending2.isPresent()
                && pending2.get().getSubmissionId().equals(taxReturnSubmissionReceipt3.getSubmissionId()));

        assertEquals(2, taxReturnSubmissionRepo.count());
        Optional<String> submissionId1 = taxReturnSubmissionRepo.getLatestSubmissionIdByTaxReturnId(
                taxReturnSubmissionReceipt1.getTaxReturnId());
        assertTrue(
                submissionId1.isPresent() && submissionId1.get().equals(taxReturnSubmissionReceipt1.getSubmissionId()));
        Optional<String> submissionId2 = taxReturnSubmissionRepo.getLatestSubmissionIdByTaxReturnId(
                taxReturnSubmissionReceipt3.getTaxReturnId());
        assertTrue(
                submissionId2.isPresent() && submissionId2.get().equals(taxReturnSubmissionReceipt3.getSubmissionId()));
    }

    @Test
    public void handleSubmissionConfirmationMessage_noSubmittedReturns() {
        SubmissionConfirmationV2Handler handler =
                new SubmissionConfirmationV2Handler(pendingRepo, taxReturnSubmissionRepo, statusProperties);

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(UUID.randomUUID(), "submissionId1", "receiptId1", new Date());
        SubmissionConfirmationPayloadV2Entry entry1 = new SubmissionConfirmationPayloadV2Entry(
                taxReturnSubmissionReceipt1, SubmissionEventTypeEnum.FAILED, Map.of());

        List<SubmissionConfirmationPayloadV2Entry> entries = List.of(entry1);

        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV2(entries);
        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> queueMessage =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders()
                                .addHeader(
                                        MessageHeaderAttribute.VERSION,
                                        SubmissionConfirmationMessageVersion.V2.getVersion()));

        handler.handleSubmissionConfirmationMessage(queueMessage);

        assertEquals(0, pendingRepo.count());
        assertEquals(0, taxReturnSubmissionRepo.count());
    }
}
