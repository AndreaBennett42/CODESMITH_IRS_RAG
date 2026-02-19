package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.TaxReturnService;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;

@Service
@Slf4j
@Transactional
public class PublishDispatchMessageEventHandler implements S3NotificationEventHandler {
    private final TaxReturnService taxReturnService;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepository;
    private static final String PAYLOAD_KEY = "submissionIds";
    private static final int DB_QUERY_BATCH_SIZE = 50;

    public PublishDispatchMessageEventHandler(
            TaxReturnService taxReturnService, TaxReturnSubmissionRepository taxReturnSubmissionRepository) {
        this.taxReturnService = taxReturnService;
        this.taxReturnSubmissionRepository = taxReturnSubmissionRepository;
    }

    @Override
    public void handleNotificationEvent(JsonNode payload) {
        publishDispatchMessages(payload);
    }

    private void publishDispatchMessages(JsonNode payload) {
        List<String> submissionIds = new ArrayList<>();

        if (payload.get(PAYLOAD_KEY) == null) {
            log.error("Unable to parse submissionIds from S3 Notification Event payload");
            return;
        }

        Iterator<JsonNode> submissionIdsArray = payload.get(PAYLOAD_KEY).elements();
        submissionIdsArray.forEachRemaining(submissionId -> submissionIds.add(submissionId.asText()));

        if (submissionIds.isEmpty()) {
            log.error("Unable to parse submissionIds from S3 Notification Event payload - empty submissionIds element");
            return;
        }

        // Maintain a set of unprocessed submission ids for logging purposes
        Set<String> unProcessedSubmissionIds = new HashSet<>(submissionIds);

        try {
            for (int i = 0; i < submissionIds.size(); i += DB_QUERY_BATCH_SIZE) {
                int end = Math.min(i + DB_QUERY_BATCH_SIZE, submissionIds.size());
                List<String> submissionIdsSubList = submissionIds.subList(i, end);

                List<TaxReturnSubmission> taxReturnSubmissions =
                        taxReturnSubmissionRepository.findAllBySubmissionIds(submissionIdsSubList);

                taxReturnSubmissions.forEach(taxReturnSubmission -> {
                    // stub out enqueueing

                    // mark submission as processed
                    unProcessedSubmissionIds.remove(taxReturnSubmission.getSubmissionId());
                });
            }
        } catch (Exception e) {
            log.error("Error processing publish_dispatch_queue_messages S3 Event", e);
        }

        if (unProcessedSubmissionIds.isEmpty()) {
            log.info(
                    "Successfully dispatched all of the specified tax return submissions. Total: {}",
                    submissionIds.size());
        } else {
            log.error(
                    "Unable to dispatch tax return submissions with the following submissionIds: {}",
                    unProcessedSubmissionIds);
        }
    }
}
