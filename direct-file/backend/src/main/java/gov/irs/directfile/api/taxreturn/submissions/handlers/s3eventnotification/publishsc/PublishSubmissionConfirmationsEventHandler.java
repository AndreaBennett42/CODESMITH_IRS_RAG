package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc;

import java.time.Instant;
import java.util.*;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.S3NotificationEventHandler;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Service
@Slf4j
@Transactional
public class PublishSubmissionConfirmationsEventHandler implements S3NotificationEventHandler {
    protected final TaxReturnSubmissionRepository taxReturnSubmissionRepository;
    protected final SubmissionConfirmationMessageService submissionConfirmationMessageService;
    protected final ObjectMapper objectMapper;

    protected static final String SUBMISSION_IDS_KEY = "submissionIds";
    protected static final String TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY =
            "taxReturnSubmissionReceiptInformations";
    protected static final int MIN_BATCH_SIZE = 1;
    protected static final int MAX_BATCH_SIZE = 250;
    protected static final int DEFAULT_BATCH_SIZE = 100;

    public PublishSubmissionConfirmationsEventHandler(
            TaxReturnSubmissionRepository taxReturnSubmissionRepository,
            SubmissionConfirmationMessageService submissionConfirmationMessageService,
            ObjectMapper objectMapper) {
        this.taxReturnSubmissionRepository = taxReturnSubmissionRepository;
        this.submissionConfirmationMessageService = submissionConfirmationMessageService;
        this.objectMapper = objectMapper;
    }

    @Override
    public void handleNotificationEvent(JsonNode payload) {
        if (!payload.has(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY)) {
            log.error("JSON must have submissionIds");
            return;
        }

        int batchSize = DEFAULT_BATCH_SIZE;
        if (payload.has("batchSize")) {
            batchSize = payload.get("batchSize").asInt();
        }
        batchSize = Math.min(Math.max(MIN_BATCH_SIZE, batchSize), MAX_BATCH_SIZE);

        List<TaxReturnSubmissionIdAndReceiptId> taxReturnSubmissionIdAndReceiptIds = objectMapper.convertValue(
                payload.get(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY), new TypeReference<>() {});

        // Partition list into buckets of batchSize elements
        List<List<TaxReturnSubmissionIdAndReceiptId>> partitions =
                partitionList(taxReturnSubmissionIdAndReceiptIds, batchSize);
        for (List<TaxReturnSubmissionIdAndReceiptId> partition : partitions) {
            publishSubmissionConfirmations(partition);
        }
    }

    protected void publishSubmissionConfirmations(
            List<TaxReturnSubmissionIdAndReceiptId> taxReturnSubmissionIdAndReceiptIds) {
        try {
            List<String> submissionIds = taxReturnSubmissionIdAndReceiptIds.stream()
                    .map(TaxReturnSubmissionIdAndReceiptId::getSubmissionId)
                    .toList();

            List<TaxReturnSubmission> taxReturnSubmissions =
                    taxReturnSubmissionRepository.findAllBySubmissionIds(submissionIds);

            if (taxReturnSubmissions.size() < taxReturnSubmissionIdAndReceiptIds.size()) {
                log.error("Unable to find all TaxReturnSubmissions associated with the given submissionIds");
            }

            Map<String, String> receiptIdFromSubmissionIdMap = new HashMap<>();
            taxReturnSubmissionIdAndReceiptIds.forEach(
                    t -> receiptIdFromSubmissionIdMap.put(t.getSubmissionId(), t.getReceiptId()));

            List<SubmissionConfirmationPayloadV2Entry> entries = taxReturnSubmissions.stream()
                    .map(taxReturnSubmission -> {
                        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt = new TaxReturnSubmissionReceipt(
                                taxReturnSubmission.getTaxReturnId(),
                                taxReturnSubmission.getSubmissionId(),
                                receiptIdFromSubmissionIdMap.get(taxReturnSubmission.getSubmissionId()),
                                Date.from(Instant.now()));

                        return new SubmissionConfirmationPayloadV2Entry(
                                taxReturnSubmissionReceipt, SubmissionEventTypeEnum.SUBMITTED, null);
                    })
                    .toList();

            submissionConfirmationMessageService.publishSubmissionConfirmationPayloadV2(entries);

            log.info(
                    "Successfully published for taxReturnSubmissionReceiptInformations: {}",
                    taxReturnSubmissionIdAndReceiptIds);
        } catch (Exception e) {
            log.error("Error processing publish_submission_confirmations S3 event", e);
        }
    }

    protected List<List<TaxReturnSubmissionIdAndReceiptId>> partitionList(
            List<TaxReturnSubmissionIdAndReceiptId> taxReturnSubmissionIdAndReceiptIds, int batchSize) {
        List<List<TaxReturnSubmissionIdAndReceiptId>> partitions = new ArrayList<>();
        for (int i = 0; i < taxReturnSubmissionIdAndReceiptIds.size(); i += batchSize) {
            List<TaxReturnSubmissionIdAndReceiptId> partition = taxReturnSubmissionIdAndReceiptIds.subList(
                    i, Math.min(i + batchSize, taxReturnSubmissionIdAndReceiptIds.size()));
            partitions.add(partition);
        }
        return partitions;
    }
}
