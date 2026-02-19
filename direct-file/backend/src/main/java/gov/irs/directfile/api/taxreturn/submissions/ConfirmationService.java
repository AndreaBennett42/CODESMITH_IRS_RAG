package gov.irs.directfile.api.taxreturn.submissions;

import java.util.*;

import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.slf4j.spi.LoggingEventBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import gov.irs.factgraph.Graph;

import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.taxreturn.*;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.models.LepLanguage;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.SubmissionEventFailureCategoryEnum;
import gov.irs.directfile.models.message.SubmissionEventFailureDetailEnum;
import gov.irs.directfile.models.message.SubmissionEventFailureInterface;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Slf4j
@Service
@SuppressWarnings({"PMD.ExcessiveParameterList", "PMD.UnusedLocalVariable"})
public class ConfirmationService {
    private final TaxReturnRepository taxReturnRepo;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepo;
    private final TaxReturnService taxReturnService;
    private final SendEmailQueueService sendEmailQueueService;
    private final FactGraphService factGraphService;
    private final SubmissionEventRepository submissionEventRepository;

    private final int maxDispatchEnqueueAttempts;
    private final StatusResponseBodyCacheService statusResponseBodyCacheService;

    public ConfirmationService(
            final TaxReturnRepository taxReturnRepo,
            final TaxReturnSubmissionRepository taxReturnSubmissionRepo,
            final TaxReturnService taxReturnService,
            final SendEmailQueueService sendEmailQueueService,
            final FactGraphService factGraphService,
            final SubmissionEventRepository submissionEventRepository,
            @Value("${direct-file.max_dispatch_enqueue_attempts}") int maxDispatchEnqueueAttempts,
            final StatusResponseBodyCacheService statusResponseBodyCacheService) {
        this.taxReturnRepo = taxReturnRepo;
        this.taxReturnSubmissionRepo = taxReturnSubmissionRepo;
        this.taxReturnService = taxReturnService;
        this.sendEmailQueueService = sendEmailQueueService;
        this.factGraphService = factGraphService;
        this.submissionEventRepository = submissionEventRepository;
        this.maxDispatchEnqueueAttempts = maxDispatchEnqueueAttempts;
        this.statusResponseBodyCacheService = statusResponseBodyCacheService;
    }

    @Transactional
    public void handleSubmissionConfirmations(List<TaxReturnSubmissionReceipt> taxReturnSubmissionReceipts) {
        List<TaxReturnSubmission> taxReturnSubmissionsToUpdate = new ArrayList<>();
        LoggingEventBuilder builder = log.atInfo();
        List<UUID> taxReturnIds = taxReturnSubmissionReceipts.stream()
                .map(TaxReturnSubmissionReceipt::getTaxReturnId)
                .toList();
        List<TaxReturnSubmission> taxReturnSubmissions =
                taxReturnSubmissionRepo.findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(taxReturnIds);
        Map<UUID, TaxReturnSubmission> taxReturnIdToTaxReturnSubmission = new HashMap<>();
        taxReturnSubmissions.forEach(trs -> {
            taxReturnIdToTaxReturnSubmission.put(trs.getTaxReturnId(), trs);
        });
        Map<TaxReturnSubmissionReceipt, TaxReturnSubmission> receiptToTaxReturnSubmissionMap =
                mapReceiptIdsToTaxReturnSubmission(taxReturnSubmissionReceipts, taxReturnIdToTaxReturnSubmission);

        receiptToTaxReturnSubmissionMap.forEach((taxReturnSubmissionReceipt, taxReturnSubmission) -> {
            UUID taxReturnId = taxReturnSubmissionReceipt.getTaxReturnId();
            String mefSubmissionId = taxReturnSubmissionReceipt.getSubmissionId();
            if (taxReturnSubmissionReceipt.getReceiptId() == null
                    || taxReturnSubmissionReceipt.getReceiptId().isBlank()) {
                log.error("receipt ID for tax return ID {} is blank", taxReturnId);
            }
            // only update the TaxReturnSubmission if the receipt ID is not already set
            // a previously set receiptId would indicate a duplicate submission, in which case we ignore the event
            if (taxReturnSubmission.getReceiptId() == null
                    || taxReturnSubmission.getReceiptId().isBlank()) {
                taxReturnSubmission.setReceiptId(taxReturnSubmissionReceipt.getReceiptId());
                taxReturnSubmission.setSubmissionId(mefSubmissionId);
                taxReturnSubmission.setSubmissionReceivedAt(taxReturnSubmissionReceipt.getSubmissionReceivedAt());
                taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
                taxReturnSubmissionsToUpdate.add(taxReturnSubmission);
                builder.addKeyValue(AuditLogElement.TAX_RETURN_ID.toString(), taxReturnId.toString())
                        .addKeyValue(AuditLogElement.MEF_SUBMISSION_ID.toString(), mefSubmissionId)
                        .addKeyValue(
                                AuditLogElement.DETAIL.toString(),
                                "Received and processed submission confirmation message")
                        .log();
            }
        });
        log.info(String.format(
                "Updating %s tax return submission objects for submission events",
                taxReturnSubmissionsToUpdate.size()));
        taxReturnSubmissionRepo.saveAll(taxReturnSubmissionsToUpdate);
    }

    @Transactional
    public void handleStatusChangeEvents(List<SubmissionStatusesMessage> submissionStatuses) {
        Map<String, List<TaxReturnSubmission>> statusTaxReturnSubmissionMap = addSubmissionEvents(submissionStatuses);
        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> messagesToSend =
                createStatusChangeMessages(statusTaxReturnSubmissionMap);
        if (!messagesToSend
                        .getOrDefault(HtmlTemplate.ACCEPTED, new ArrayList<>())
                        .isEmpty()
                || !messagesToSend
                        .getOrDefault(HtmlTemplate.REJECTED, new ArrayList<>())
                        .isEmpty()) {
            // log the taxReturnId's that we intend to send an email about
            List<UUID> taxReturnIds = new ArrayList<>();
            messagesToSend
                    .getOrDefault(HtmlTemplate.ACCEPTED, new ArrayList<>())
                    .forEach(i -> taxReturnIds.add(i.getTaxReturnId()));
            messagesToSend
                    .getOrDefault(HtmlTemplate.REJECTED, new ArrayList<>())
                    .forEach(i -> taxReturnIds.add(i.getTaxReturnId()));
            log.atInfo()
                    .addKeyValue("numberOfEmailsToSend", taxReturnIds.size())
                    .addKeyValue("taxReturnIds", taxReturnIds)
                    .log("Enqueueing message to send emails");

            sendEmailQueueService.enqueue(messagesToSend);
        }
    }

    @Transactional
    public void handleSubmissionFailures(List<SubmissionConfirmationPayloadV2Entry> entries) {
        // map to associate submissionId with its error detail (metadata), used later in building a FAILED
        // submissionEvent object
        final Map<String, Map<String, SubmissionEventFailureInterface>> metadataMaps = new HashMap<>();
        Map<UUID, String> taxReturnIdSubmissionIdMap = new HashMap<>();
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();
        entries.forEach(entry -> {
            Map<String, SubmissionEventFailureInterface> metadataMap = new HashMap<>();
            metadataMap.put(
                    "failureCategory",
                    SubmissionEventFailureCategoryEnum.getEnum(
                            entry.getMetadata().get("failureCategory")));
            metadataMap.put(
                    "failureDetail",
                    SubmissionEventFailureDetailEnum.getEnum(entry.getMetadata().get("failureDetail")));
            metadataMaps.put(entry.getTaxReturnSubmissionReceipt().getSubmissionId(), metadataMap);
            taxReturnIdSubmissionIdMap.put(
                    entry.getTaxReturnSubmissionReceipt().getTaxReturnId(),
                    entry.getTaxReturnSubmissionReceipt().getSubmissionId());
        });

        // get taxreturn_submissions records via the keys of metadataMaps (i.e. the submissionIds)
        List<TaxReturnSubmission> taxReturnSubmissions = taxReturnSubmissionRepo.findAllBySubmissionIds(
                metadataMaps.keySet().stream().toList());
        // add a FAILED submissionEvent to each taxReturnSubmission object
        taxReturnSubmissions.forEach(taxReturnSubmission -> {
            taxReturnSubmission.addSubmissionEvent(
                    SubmissionEventTypeEnum.FAILED,
                    (SubmissionEventFailureCategoryEnum) metadataMaps
                            .get(taxReturnSubmission.getSubmissionId())
                            .get("failureCategory"),
                    (SubmissionEventFailureDetailEnum) metadataMaps
                            .get(taxReturnSubmission.getSubmissionId())
                            .get("failureDetail"));
            taxReturnTaxReturnSubmissionMap.put(taxReturnSubmission.getTaxReturnId(), taxReturnSubmission);
        });

        log.info("Updating tax return submission with failed status");
        // save updated taxReturnSubmission objects (results in saving the FAILED submissionEvent object as well)
        taxReturnSubmissionRepo.saveAll(taxReturnSubmissions);

        // Clear the status cache for any failed submissions we update so the status endpoint will determine
        // the status again.
        clearStatusReturnBodyCache(taxReturnSubmissions);

        enqueueDispatchForFailedSubmissions(taxReturnIdSubmissionIdMap, taxReturnTaxReturnSubmissionMap);
    }

    protected void enqueueDispatchForFailedSubmissions(
            Map<UUID, String> taxReturnIdSubmissionIdMap,
            Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap) {
        List<TaxReturn> taxReturns = taxReturnRepo.findAllByTaxReturnIds(
                taxReturnIdSubmissionIdMap.keySet().stream().toList());
        taxReturns.forEach(taxReturn -> {
            UUID taxReturnId = taxReturn.getId();
            String submissionId = taxReturnIdSubmissionIdMap.get(taxReturnId);
            TaxReturnSubmission taxReturnSubmission = taxReturnTaxReturnSubmissionMap.get(taxReturnId);
            int totalFailedEvents = submissionEventRepository.countFailedEvents(taxReturnSubmission.getId());
            Map<UUID, TaxReturnSubmission> submissionMap = new HashMap<>();
            if (totalFailedEvents <= this.maxDispatchEnqueueAttempts) {
                log.info("{} has {} failed submission events and can enqueue dispatch", taxReturnId, totalFailedEvents);
                taxReturnService.stubEnqueueDispatch();
            } else {
                log.info(
                        "{} has {} failed submission events. Not enqueueing dispatch and instead sending post-submission error email",
                        taxReturnId,
                        totalFailedEvents);
                submissionMap.put(taxReturnId, taxReturnSubmission);
                enqueuePostSubmissionErrorEmailForFailedSubmissions(submissionMap);
            }
        });
    }

    protected void enqueuePostSubmissionErrorEmailForFailedSubmissions(
            Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap) {
        enqueueStatusChangeEmail(taxReturnTaxReturnSubmissionMap, HtmlTemplate.POST_SUBMISSION_ERROR);
    }

    public void enqueueErrorResolutionEmail(Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap) {
        enqueueStatusChangeEmail(taxReturnTaxReturnSubmissionMap, HtmlTemplate.ERROR_RESOLVED);
    }

    public void enqueueStatusChangeEmail(
            Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap, HtmlTemplate template) {
        log.info("Calling enqueueEmail for template {}", template.toString());
        Map<String, List<TaxReturnSubmission>> statusTaxReturnSubmissionMap = new HashMap<>();
        statusTaxReturnSubmissionMap.put(
                template.toString().toLowerCase(),
                taxReturnTaxReturnSubmissionMap.values().stream().toList());
        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> messagesToSend =
                createStatusChangeMessages(statusTaxReturnSubmissionMap);
        sendEmailQueueService.enqueue(messagesToSend);
    }

    @Transactional
    protected Map<String, List<TaxReturnSubmission>> addSubmissionEvents(
            List<SubmissionStatusesMessage> submissionStatuses) {

        /**
         * 1. Pre-populate the return object with an empty list
         * for each status in the Message.
         * 2. To de-duplicate submission statuses,
         * store a Map<SubmissionId, SubmissionStatuses>.
         *
         * If a given submission id has 1 accepted, and 1 rejected event - this represents
         * a case where the return was Accepted, and later rejected due to a duplicate submission.
         *
         * The submission statuses in the map for such a return would be (accepted, rejected).
         *
         * If a given submission id has 2 rejected events - this represents the case where
         * the return was Rejected based on the content of the return, then rejected again due to
         * being a duplicate.
         *
         * The submission status in the map for such a return would be (rejected), because we store
         * them in a set.
         *
         * OUTPUT:
         * statusTaxReturnSubmissionMap
         * {
         *    "ACCEPTED": [],
         *    "REJECTED": [],
         * }
         *
         * submissionIdToStatuses (where each ["Accepted"] is actually a set, not a list)
         * {
         *     "sub_1" : ["Accepted", "Rejected],
         *     "sub_2" : ["Accepted"],
         *     "sub_3" : ["Accepted"],
         *     "sub_4" : ["Accepted"],
         *     "sub_5" : ["Rejected"],
         *     "sub_6" : ["Rejected"],
         *     "sub_7" : ["Rejected"],
         * }
         * */
        Map<String, List<TaxReturnSubmission>> statusTaxReturnSubmissionMap = new HashMap<>();
        Map<String, Set<SubmissionEventTypeEnum>> submissionIdToStatuses = new HashMap<>();
        for (SubmissionStatusesMessage submissionStatusMessage : submissionStatuses) {
            String status = submissionStatusMessage.status().name();
            statusTaxReturnSubmissionMap.put(status, new ArrayList<>());
            SubmissionEventTypeEnum submissionStatus = SubmissionEventTypeEnum.getEnum(status);
            for (String submissionId : submissionStatusMessage.submissionIds()) {
                Set<SubmissionEventTypeEnum> statusesForSubmission =
                        submissionIdToStatuses.getOrDefault(submissionId, new HashSet<>());
                statusesForSubmission.add(submissionStatus);
                submissionIdToStatuses.put(submissionId, statusesForSubmission);
            }
        }

        /**
         * 3. Get all the taxReturnSubmissions associated with the Submission Ids
         * */
        List<String> submissionIds = submissionIdToStatuses.keySet().stream().toList();
        List<UUID> taxReturnSubmissionIds = taxReturnSubmissionRepo.findIdBySubmissionId(submissionIds);
        List<TaxReturnSubmission> taxReturnSubmissionsWithRejectedEvents =
                taxReturnSubmissionRepo.findAllWithTerminalEventsByTaxReturnSubmissionIdByEventType(
                        taxReturnSubmissionIds,
                        SubmissionEventTypeEnum.REJECTED.toString().toLowerCase());
        List<TaxReturnSubmission> taxReturnSubmissionsWithoutTerminalEvents =
                taxReturnSubmissionRepo.findAllWithoutTerminalEventsByTaxReturnSubmissionId(taxReturnSubmissionIds);
        List<TaxReturnSubmission> taxReturnSubmissionsToUpdate = new ArrayList<>();

        /**
         * For each taxReturnSubmissionsWithoutTerminalEvents:
         * 1. Check if the submission already has an accepted or rejected event.
         * 2. If one does not exist
         *     - If one of the status changes is accepted, save the TaxReturnSubmission as "accepted"
         *     - If all the statues are rejected, save the TaxReturnSubmission event as "rejected"
         * */
        taxReturnSubmissionsWithoutTerminalEvents.forEach(trs -> {
            taxReturnSubmissionsToUpdate.add(trs);
            if (submissionIdToStatuses.get(trs.getSubmissionId()).contains(SubmissionEventTypeEnum.ACCEPTED)) {
                trs.addSubmissionEvent(SubmissionEventTypeEnum.ACCEPTED);
                statusTaxReturnSubmissionMap
                        .get(SubmissionEventTypeEnum.ACCEPTED.getEventType().toUpperCase())
                        .add(trs);
            } else {
                trs.addSubmissionEvent(SubmissionEventTypeEnum.REJECTED);
                statusTaxReturnSubmissionMap
                        .get(SubmissionEventTypeEnum.REJECTED.getEventType().toUpperCase())
                        .add(trs);
            }
        });

        /**
         * For each taxReturnSubmissionsWithRejectedEvents:
         * 1. Check if the submission already has an accepted or rejected event.
         * 2. If one does not exist
         *     - If one of the status changes is accepted, save the TaxReturnSubmission as "accepted"
         *     - If all the statues are rejected, save the TaxReturnSubmission event as "rejected"
         * */
        taxReturnSubmissionsWithRejectedEvents.forEach(trs -> {
            // Case where a submission has been rejected already, but the new status change is accepted.
            if (submissionIdToStatuses.get(trs.getSubmissionId()).contains(SubmissionEventTypeEnum.ACCEPTED)) {
                trs.addSubmissionEvent(SubmissionEventTypeEnum.ACCEPTED);
                statusTaxReturnSubmissionMap
                        .get(SubmissionEventTypeEnum.ACCEPTED.getEventType().toUpperCase())
                        .add(trs);

                taxReturnSubmissionsToUpdate.add(trs);
            } else {
                // Case where a submission has been rejected already, and receives another rejection.
                MDC.put(
                        gov.irs.directfile.audit.AuditLogElement.taxReturnId.toString(),
                        trs.getTaxReturnId().toString());
                MDC.put(gov.irs.directfile.audit.AuditLogElement.mefSubmissionId.toString(), trs.getSubmissionId());
                log.warn(
                        "Received a status change event after tax return submission has already been processed -- tax return id {}, submission id {}",
                        trs.getTaxReturnId(),
                        trs.getSubmissionId());
                MDC.clear();
            }
        });

        taxReturnSubmissionRepo.saveAll(taxReturnSubmissionsToUpdate);

        // Clear the status cache for any accepted/rejected submissions we update so the status endpoint
        // will determine the status again.
        clearStatusReturnBodyCache(taxReturnSubmissionsToUpdate);

        return statusTaxReturnSubmissionMap;
    }

    public void enqueueUnsubmittedReturnsMessages(List<TaxReturn> taxReturns) {
        List<SendEmailQueueMessageBody> unsubmittedTaxReturnMesssageList = unsubmittedTaxReturnMessages(taxReturns);
        sendEmailQueueService.enqueue(Map.of(HtmlTemplate.REMINDER_SUBMIT, unsubmittedTaxReturnMesssageList));
    }

    protected List<SendEmailQueueMessageBody> unsubmittedTaxReturnMessages(List<TaxReturn> taxReturns) {
        List<SendEmailQueueMessageBody> messages = new ArrayList<>();
        for (TaxReturn taxreturn : taxReturns) {
            Graph graph = factGraphService.getGraph(taxreturn.getFacts());
            String email = taxReturnService.getFactGraphEmail(graph);
            if (email != null) {
                messages.add(createUnsubmittedReturnReminderMessage(taxreturn, graph, email));
            } else {
                log.error(String.format(
                        "Could not find email in fact graph for tax return ID %s and will not send email message",
                        taxreturn.getId()));
            }
        }
        return messages;
    }

    protected Map<HtmlTemplate, List<SendEmailQueueMessageBody>> createStatusChangeMessages(
            Map<String, List<TaxReturnSubmission>> statusTaxReturnSubmissionMap) {
        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> messagesToSend = new HashMap<>();

        statusTaxReturnSubmissionMap.forEach((status, submissions) -> {
            List<SendEmailQueueMessageBody> messages = new ArrayList<>();

            submissions.forEach(taxReturnSubmission -> {
                Graph graph = factGraphService.getGraph(taxReturnSubmission.getFacts());
                String email = taxReturnService.getFactGraphEmail(graph);
                if (email != null) {
                    messages.add(createStatusChangeMessage(taxReturnSubmission, graph, email));
                } else {
                    log.error(String.format(
                            "Could not find email in fact graph for tax return ID %s and will not send email message",
                            taxReturnSubmission.getTaxReturnId()));
                }
            });
            messagesToSend.put(HtmlTemplate.valueOf(status.toUpperCase()), messages);
        });
        return messagesToSend;
    }

    protected SendEmailQueueMessageBody createStatusChangeMessage(
            TaxReturnSubmission taxReturnSubmission, Graph graph, String email) {

        UUID userId = null;
        Optional<User> userOptional =
                taxReturnSubmission.getTaxReturn().getOwners().stream().findFirst();
        if (userOptional.isPresent()) {
            userId = userOptional.get().getId();
        }

        final LepLanguage lepLanguage = LepLanguage.fromFactGraph(graph);

        SendEmailQueueMessageBody sendEmailQueueMessageBody = new SendEmailQueueMessageBody(
                email,
                LepLanguage.getDefaultIfNotEnabled(lepLanguage).toCode(),
                taxReturnSubmission.getTaxReturnId(),
                taxReturnSubmission.getSubmissionId(),
                userId);
        log.info(String.format("Preparing message for taxReturnId %s ", taxReturnSubmission.getTaxReturnId()));
        return sendEmailQueueMessageBody;
    }

    protected SendEmailQueueMessageBody createUnsubmittedReturnReminderMessage(
            TaxReturn taxReturn, Graph graph, String email) {

        UUID userId = null;
        Optional<User> userOptional = taxReturn.getOwners().stream().findFirst();
        if (userOptional.isPresent()) {
            userId = userOptional.get().getId();
        }

        final LepLanguage lepLanguage = LepLanguage.fromFactGraph(graph);

        SendEmailQueueMessageBody sendEmailQueueMessageBody = new SendEmailQueueMessageBody(
                email,
                LepLanguage.getDefaultIfNotEnabled(lepLanguage).toCode(),
                taxReturn.getId(),
                null, // This return has not been submitted so it does not have a submission id
                userId);
        log.info(String.format("Preparing message for taxReturnId %s ", taxReturn.getId()));
        return sendEmailQueueMessageBody;
    }

    private static Map<TaxReturnSubmissionReceipt, TaxReturnSubmission> mapReceiptIdsToTaxReturnSubmission(
            List<TaxReturnSubmissionReceipt> taxReturnSubmissionReceipts,
            Map<UUID, TaxReturnSubmission> taxReturnIdToTaxReturnSubmissionId) {
        Map<TaxReturnSubmissionReceipt, TaxReturnSubmission> receiptToTaxReturnSubmissionIdMap = new HashMap<>();
        taxReturnSubmissionReceipts.forEach(trsr -> {
            TaxReturnSubmission taxReturnSubmission = taxReturnIdToTaxReturnSubmissionId.get(trsr.getTaxReturnId());
            if (taxReturnSubmission != null) {
                receiptToTaxReturnSubmissionIdMap.put(trsr, taxReturnSubmission);
            }
        });

        return receiptToTaxReturnSubmissionIdMap;
    }

    protected void clearStatusReturnBodyCache(List<TaxReturnSubmission> taxReturnSubmissions) {
        for (TaxReturnSubmission taxReturnSubmission : taxReturnSubmissions) {
            log.info("Cache clear of StatusResponseBody for submission ID {}", taxReturnSubmission.getSubmissionId());
            statusResponseBodyCacheService.clearKey(taxReturnSubmission.getSubmissionId());
        }
    }
}
