package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.KeysetScrollPosition;
import org.springframework.data.domain.Limit;
import org.springframework.data.domain.ScrollPosition;
import org.springframework.data.domain.Window;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.stateapi.domain.export.FilingObligationState;
import gov.irs.directfile.api.taxreturn.SimpleTaxReturnProjection;
import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.api.taxreturn.submissions.ReminderEmailCacheService;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Service
@Slf4j
@Transactional
public class ReminderEmailHandler implements S3NotificationEventHandler {

    private final ConfirmationService confirmationService;

    private final TaxReturnSubmissionRepository taxReturnSubmissionRepository;
    private final TaxReturnRepository taxReturnRepository;
    final ReminderEmailCacheService reminderEmailCacheService;

    String ID_ARR_KEY = "ids";
    String SUBMIT_VALUE = "submit";
    String RESUBMIT_VALUE = "resubmit";
    String ACCEPTED_VALUE = "accepted";
    String REJECTED_VALUE = "rejected";
    String SUBMITTED_VALUE = "submitted";
    String TECHNICAL_ERROR_VALUE = "technical_error";
    String PRE_SUBMISSION_ERROR = "pre_submission_error";
    String ERROR_RESOLVED_VALUE = "error_resolved";
    String REMINDER_CATEGORY_KEY = "reminder_category_key";
    private final Map<String, HtmlTemplate> TEMPLATE_MAP = new HashMap<>();
    private final Map<String, SubmissionEventTypeEnum> SUBMISSION_EVENT_TYPE_MAP = new HashMap<>();

    public ReminderEmailHandler(
            ConfirmationService confirmationService,
            TaxReturnSubmissionRepository taxReturnSubmissionRepository,
            TaxReturnRepository taxReturnRepository) {
        this.confirmationService = confirmationService;
        this.taxReturnSubmissionRepository = taxReturnSubmissionRepository;
        this.taxReturnRepository = taxReturnRepository;
        this.reminderEmailCacheService = new ReminderEmailCacheService();
        initTemplateMap();
    }

    private void initTemplateMap() {

        for (String stateAbbreviation : FilingObligationState.abbreviations()) {
            TEMPLATE_MAP.put(stateAbbreviation, HtmlTemplate.REMINDER_STATE);
            SUBMISSION_EVENT_TYPE_MAP.put(stateAbbreviation, SubmissionEventTypeEnum.REMINDER_STATE_TAX);
        }
        TEMPLATE_MAP.put(SUBMIT_VALUE, HtmlTemplate.REMINDER_SUBMIT);
        TEMPLATE_MAP.put(RESUBMIT_VALUE, HtmlTemplate.REMINDER_RESUBMIT);
        TEMPLATE_MAP.put(ACCEPTED_VALUE, HtmlTemplate.ACCEPTED);
        TEMPLATE_MAP.put(REJECTED_VALUE, HtmlTemplate.REJECTED);
        TEMPLATE_MAP.put(SUBMITTED_VALUE, HtmlTemplate.SUBMITTED);
        TEMPLATE_MAP.put(TECHNICAL_ERROR_VALUE, HtmlTemplate.POST_SUBMISSION_ERROR);
        TEMPLATE_MAP.put(PRE_SUBMISSION_ERROR, HtmlTemplate.PRE_SUBMISSION_ERROR);
        TEMPLATE_MAP.put(ERROR_RESOLVED_VALUE, HtmlTemplate.ERROR_RESOLVED);
        SUBMISSION_EVENT_TYPE_MAP.put(SUBMIT_VALUE, SubmissionEventTypeEnum.REMINDER_SUBMIT);
        SUBMISSION_EVENT_TYPE_MAP.put(RESUBMIT_VALUE, SubmissionEventTypeEnum.REMINDER_RESUBMIT);
        SUBMISSION_EVENT_TYPE_MAP.put(ACCEPTED_VALUE, SubmissionEventTypeEnum.ACCEPTED);
        SUBMISSION_EVENT_TYPE_MAP.put(REJECTED_VALUE, SubmissionEventTypeEnum.REJECTED);
        SUBMISSION_EVENT_TYPE_MAP.put(SUBMITTED_VALUE, SubmissionEventTypeEnum.SUBMITTED);
        SUBMISSION_EVENT_TYPE_MAP.put(TECHNICAL_ERROR_VALUE, SubmissionEventTypeEnum.POST_SUBMISSION_ERROR);
        SUBMISSION_EVENT_TYPE_MAP.put(PRE_SUBMISSION_ERROR, SubmissionEventTypeEnum.PRE_SUBMISSION_ERROR);
        SUBMISSION_EVENT_TYPE_MAP.put(ERROR_RESOLVED_VALUE, SubmissionEventTypeEnum.ERROR_RESOLVED);
    }

    private boolean hasDateRange(JsonNode payload) {
        return payload.has("startDate") && payload.has("endDate");
    }

    @Override
    public void handleNotificationEvent(JsonNode payload) {
        String reminderCategoryKey = payload.get(REMINDER_CATEGORY_KEY).asText();
        if (reminderCategoryKey.isBlank() || reminderCategoryKey.isEmpty()) {
            throw new IllegalArgumentException(
                    "reminder_category_key cannot be empty for ReminderEmailHandler message processing");
        }
        log.info("Handling notification event for reminderCategoryKey: {}", reminderCategoryKey);

        if (hasDateRange(payload)) {
            try {
                Date startDate = new SimpleDateFormat("yyyy-MM-dd", Locale.US)
                        .parse(payload.get("startDate").asText());
                Date endDate = new SimpleDateFormat("yyyy-MM-dd", Locale.US)
                        .parse(payload.get("endDate").asText());

                if (SUBMIT_VALUE.equals(reminderCategoryKey)) {
                    loadReminderToSubmitEmailsIntoCache(startDate, endDate);
                }
            } catch (ParseException e) {
                log.error("unable to parse json");
            }
        } else {
            Iterator<JsonNode> idsArr = payload.get(ID_ARR_KEY).elements();
            putTaxReturnIdsIntoCache(idsArr, reminderCategoryKey);
        }
    }

    public void putTaxReturnIdsIntoCache(Iterator<JsonNode> idsArr, String reminderCategoryKey) {
        idsArr.forEachRemaining(id -> {
            UUID taxReturnId = UUID.fromString(id.asText());
            reminderEmailCacheService.put(taxReturnId, taxReturnId, reminderCategoryKey);
        });
    }

    public Optional<Map<String, List<UUID>>> getNextBatchToProcess() {
        return reminderEmailCacheService.getNextBatch();
    }

    @Scheduled(
            fixedDelayString = "${direct-file.s3-notification-event-cache.fixedDelayMilliseconds:}",
            initialDelay = 2000)
    public void fetchTaxReturnIdBatchFromCacheAndSendReminderEmails() {
        Optional<Map<String, List<UUID>>> optBatch = getNextBatchToProcess();
        if (optBatch.isPresent()) {
            log.info("Processing email reminder batch.");
            Map<String, List<UUID>> batch = optBatch.get();
            String reminderCategoryKey = batch.keySet().stream().findFirst().get();
            List<UUID> taxReturnIds = batch.get(reminderCategoryKey);
            sendReminderEmail(taxReturnIds, reminderCategoryKey);
            reminderEmailCacheService.evict(taxReturnIds, reminderCategoryKey);
        }
    }

    /**
     * Given a Date Range, reads in all the tax return ids for unsubmitted tax returns
     * and adds them to the ReminderEmailCache.
     *
     * Later on, the fetchTaxReturnIdBatchFromCacheAndSendReminderEmails() method should fetch them from
     * the cache and send the emails out.
     *
     *
     * */
    protected void loadReminderToSubmitEmailsIntoCache(Date startDate, Date endDate) {
        Limit limit = Limit.of(20);
        KeysetScrollPosition scrollPosition = ScrollPosition.keyset();
        Window<SimpleTaxReturnProjection> unsubmittedTaxReturns =
                taxReturnRepository.findByTaxYearAndSubmitTimeIsNullAndCreatedAtBetweenOrderByCreatedAtAsc(
                        limit, 2024, startDate, endDate, scrollPosition);

        do {
            // Add Tax Return Ids to the cache
            for (SimpleTaxReturnProjection taxReturn : unsubmittedTaxReturns.getContent()) {
                reminderEmailCacheService.put(taxReturn.getId(), taxReturn.getId(), SUBMIT_VALUE);
            }
            // Update the scroll position if there are returns to retrieve
            if (!unsubmittedTaxReturns.isEmpty() && unsubmittedTaxReturns.hasNext()) {
                KeysetScrollPosition updatedScrollPosition =
                        (KeysetScrollPosition) unsubmittedTaxReturns.positionAt(unsubmittedTaxReturns.size() - 1);
                unsubmittedTaxReturns =
                        taxReturnRepository.findByTaxYearAndSubmitTimeIsNullAndCreatedAtBetweenOrderByCreatedAtAsc(
                                limit, 2024, startDate, endDate, updatedScrollPosition);
            }
        } while (!unsubmittedTaxReturns.isEmpty() && unsubmittedTaxReturns.hasNext());
    }

    protected void sendReminderToSubmitEmail(List<UUID> taxReturnIds) {
        List<TaxReturn> taxReturns = taxReturnRepository.findAllByTaxReturnIds(taxReturnIds);
        confirmationService.enqueueUnsubmittedReturnsMessages(taxReturns);
    }

    protected void sendReminderEmail(List<UUID> taxReturnIds, String reminderCategoryKey) {
        if (SUBMIT_VALUE.equals(reminderCategoryKey)) {
            sendReminderToSubmitEmail(taxReturnIds);
        }

        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();
        List<TaxReturnSubmission> taxReturnSubmissionsToUpdate = new ArrayList<>();
        HtmlTemplate template = TEMPLATE_MAP.get(reminderCategoryKey);
        SubmissionEventTypeEnum submissionEventType = SUBMISSION_EVENT_TYPE_MAP.get(reminderCategoryKey);

        log.info("Fetching latest tax return submissions for {} returns", taxReturnIds.size());
        List<TaxReturnSubmission> latestTaxReturnSubmissions =
                taxReturnSubmissionRepository.findLatestTaxReturnSubmissions(taxReturnIds);
        latestTaxReturnSubmissions.forEach(trs -> {
            trs.addSubmissionEvent(submissionEventType);
            taxReturnSubmissionsToUpdate.add(trs);
            taxReturnTaxReturnSubmissionMap.put(trs.getTaxReturnId(), trs);
        });

        log.info(
                "Saving {} tax return submissions with a {} submission event",
                taxReturnSubmissionsToUpdate.size(),
                submissionEventType.getEventType());
        taxReturnSubmissionRepository.saveAll(taxReturnSubmissionsToUpdate);
        if (!taxReturnTaxReturnSubmissionMap.isEmpty()) {
            confirmationService.enqueueStatusChangeEmail(taxReturnTaxReturnSubmissionMap, template);
        }
    }
}
