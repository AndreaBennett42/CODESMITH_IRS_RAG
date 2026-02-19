package gov.irs.directfile.status.acknowledgement;

import java.math.RoundingMode;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import com.google.common.math.IntMath;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.EntityNotFoundException;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import gov.irs.mef.exception.ServiceException;
import gov.irs.mef.exception.ToolkitException;
import gov.irs.mef.services.ServiceContext;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.models.RejectedStatus;
import gov.irs.directfile.status.acknowledgement.domain.AcknowledgementStatus;
import gov.irs.directfile.status.acknowledgement.domain.Status;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.*;
import gov.irs.directfile.status.domain.Error;
import gov.irs.directfile.status.error.ErrorRepository;
import gov.irs.directfile.status.error.ToolkitErrorRepository;
import gov.irs.directfile.status.mef.client.MeFAcksMTOMClientService;
import gov.irs.directfile.status.mef.client.MeFLoginClientService;
import gov.irs.directfile.status.mef.client.MeFLogoutClientService;
import gov.irs.directfile.status.repository.PodIdentifierRepository;
import gov.irs.directfile.status.services.StatusChangeMessageService;

@SuppressFBWarnings(
        value = {"CRLF_INJECTION_LOGS", "NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@Service
@EnableScheduling
@Transactional
@Slf4j
@SuppressWarnings({
    "PMD.SimpleDateFormatNeedsLocale",
    "PMD.ExcessiveParameterList",
    "PMD.AvoidDuplicateLiterals",
    "PMD.LiteralsFirstInComparisons",
    "PMD.UselessParentheses"
})
public class AcknowledgementService {
    private final CompletedAcknowledgementRepository completedRepo;
    private final PendingAcknowledgementRepository pendingRepo;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepository;
    private final ErrorRepository errorRepo;
    private final PodIdentifierRepository podIdentifierRepository;

    private final ToolkitErrorRepository toolkitErrorRepo;
    private final StatusProperties statusProperties;
    private ServiceContext serviceContext;

    private final StatusChangeMessageService statusChangeMessageService;

    private SimpleDateFormat simpleDateFormat = new SimpleDateFormat("hh:mm:ss");

    @SuppressFBWarnings(
            value = {"EI_EXPOSE_REP2"},
            justification = "constructor injection")
    public AcknowledgementService(
            CompletedAcknowledgementRepository completedRepo,
            PendingAcknowledgementRepository pendingRepo,
            TaxReturnSubmissionRepository taxReturnSubmissionRepository,
            ErrorRepository errorRepo,
            ToolkitErrorRepository toolkitErrorRepo,
            StatusProperties statusProperties,
            StatusChangeMessageService statusChangeMessageService,
            MeFAcksMTOMClientService getAcksClientService,
            MeFLoginClientService loginClientService,
            MeFLogoutClientService logoutClientService,
            PodIdentifierRepository podIdentifierRepository) {
        this.completedRepo = completedRepo;
        this.pendingRepo = pendingRepo;
        this.taxReturnSubmissionRepository = taxReturnSubmissionRepository;
        this.errorRepo = errorRepo;
        this.toolkitErrorRepo = toolkitErrorRepo;
        this.statusProperties = statusProperties;
        this.statusChangeMessageService = statusChangeMessageService;
        this.getAcksClientService = getAcksClientService;
        this.loginClientService = loginClientService;
        this.logoutClientService = logoutClientService;
        this.podIdentifierRepository = podIdentifierRepository;
    }

    // changed from protected to public, called by TaxReturnXmlServiceImpl (which would be separated into its own
    // microservice later)
    public String getLatestSubmissionIdByTaxReturnId(UUID taxReturnId) {
        log.info("getLatestSubmissionIdByTaxReturnId for tax-return-id {}", taxReturnId);
        Optional<String> submissionId = taxReturnSubmissionRepository.getLatestSubmissionIdByTaxReturnId(taxReturnId);
        return submissionId.orElse(null);
    }

    public String getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(UUID taxReturnId) {
        log.info("Getting most relevant submission Id for taxReturnId {}", taxReturnId);

        Optional<String> latestAcceptedSubmissionId =
                taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId);

        if (latestAcceptedSubmissionId.isPresent()) {
            String submissionId = latestAcceptedSubmissionId.get();
            log.info("Using {} as the most relevant submissionId for taxReturnId {}", submissionId, taxReturnId);
            return submissionId;
        }

        log.info(
                "Falling back to retrieving the latest submissionId, regardless of status, for tax return {}",
                taxReturnId);
        return getLatestSubmissionIdByTaxReturnId(taxReturnId);
    }

    /**
     * Useful for navigating race conditions where a non-accepted submission that is more recent than an accepted
     * submission exists, where the accepted submission should be the
     *
     * @param requestedSubmissionId a submission that (ideally) belongs to a tax return submission
     * @return the latest accepted submissionId, if present
     */
    public Optional<String> getLatestAcceptedSubmissionIdOfParentTaxReturn(String requestedSubmissionId) {
        // Note: Cannot log method name alongside submissionId as it implies status
        log.info("checking for more relevant submissionId of tax return with submissionId {}", requestedSubmissionId);
        return taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId);
    }

    public Optional<Completed> getCompletedBySubmissionId(String submissionId) {
        return completedRepo.GetCompletedSubmission(submissionId);
    }

    public List<RejectedStatus> getRejectionCodesForSubmissionId(String submissionId) {
        Optional<Completed> optCompleted = getCompletedBySubmissionId(submissionId);

        // If submission ID does not have a completed record, throw an exception.
        if (optCompleted.isEmpty()) {
            throw new EntityNotFoundException("Could not find completed record for submission ID: " + submissionId);
        }

        // We have a completed submission. Convert to a List<RejectedStatus> and return the list (if it's
        // not a rejected submission, the returned list should be empty).
        Completed completed = optCompleted.get();
        return createRejectedReasonList(completed);
    }

    public AcknowledgementStatus GetAcknowledgement(UUID taxReturnId) {
        String submissionId = getLatestSubmissionIdByTaxReturnId(taxReturnId);

        return getAcknowledgementStatus(taxReturnId, submissionId);
    }

    public AcknowledgementStatus GetAcknowledgement(UUID taxReturnId, String submissionId) {
        return getAcknowledgementStatus(taxReturnId, submissionId);
    }

    private AcknowledgementStatus getAcknowledgementStatus(UUID taxReturnId, String submissionId) {
        if (submissionId == null && statusProperties.statusEndpointReturnsPendingByDefaultEnabled) {
            // if statusProperties.createPendingUponGetStatus is true, return Pending status as we previously did
            // The difference is, we are no longer saving a Pending object in the DB.
            log.atInfo()
                    .setMessage("Unable to find a submission associated with taxReturnId in TaxReturnSubmission table")
                    .addKeyValue(AuditLogElement.taxReturnId.toString(), taxReturnId)
                    .log();

            return new AcknowledgementStatus(
                    Status.Pending, CreateTranslationKey("status", "pending"), List.of(), new Date());
        }

        if (submissionId == null && !statusProperties.statusEndpointReturnsPendingByDefaultEnabled) {
            // tax return was likely not submitted to MeF
            return null;
        }
        MDC.put(AuditLogElement.mefSubmissionId.toString(), submissionId);
        MDC.put(AuditLogElement.taxReturnId.toString(), taxReturnId.toString());
        log.info(String.format("Attempting to find submission id %s in completed database", submissionId));
        MDC.clear();

        Optional<Completed> completed = completedRepo.GetCompletedSubmission(submissionId);
        // this is a completed return
        if (completed.isPresent()) {
            MDC.put(AuditLogElement.mefSubmissionId.toString(), submissionId);
            MDC.put(AuditLogElement.taxReturnId.toString(), taxReturnId.toString());
            log.info("Completed record found");
            MDC.clear();

            return getCompletedStatus(completed.get());
        }
        MDC.put(AuditLogElement.mefSubmissionId.toString(), submissionId);
        MDC.put(AuditLogElement.taxReturnId.toString(), taxReturnId.toString());
        log.info(String.format("Did not find %s in completed, checking pending", submissionId));
        MDC.clear();

        Optional<Pending> pending = pendingRepo.GetPendingSubmission(submissionId);
        if (pending.isPresent()) {
            MDC.put(AuditLogElement.mefSubmissionId.toString(), submissionId);
            MDC.put(AuditLogElement.taxReturnId.toString(), taxReturnId.toString());
            log.info(String.format("Found %s in pending, will recheck status on next pass", submissionId));
            MDC.clear();

            return new AcknowledgementStatus(
                    Status.Pending,
                    CreateTranslationKey("status", "pending"),
                    List.of(),
                    pending.get().getCreatedAt());
        }

        return null;
    }

    @Scheduled(fixedRateString = "${status.ack-poll-in-milliseconds}", initialDelay = 1000)
    public void ProcessPendingTable() {
        if (!statusProperties.isStatusPollingEnabled()) {
            log.info("Status polling is disabled in this environment");
            return;
        }
        log.info("Timer called: performing ack check");
        LookupSubmissions();
    }

    protected Iterable<Pending> getAllPending() {
        return pendingRepo.findAll();
    }

    protected Iterable<Error> getAllError() {
        return errorRepo.findAll();
    }

    protected Iterable<Completed> getAllCompleted() {
        return completedRepo.findAll();
    }

    private List<Set<Pending>> batchPendings(Iterable<Pending> pendings) {
        List<Set<Pending>> batches = new ArrayList<>();
        Set<Pending> current = new HashSet<>();
        log.info(String.format("Estimated count: %s", pendings.spliterator().estimateSize()));
        // This surprised me, but if you access it from pending every time
        // the iterator will never move to next.
        Iterator<Pending> iterator = pendings.iterator();
        while (iterator.hasNext()) {
            Pending pending = iterator.next();
            if (current.size() < 100) {
                current.add(pending);
            } else {
                log.info("100 Pendings in current batch, creating new batch");
                batches.add(current);
                current = new HashSet<>();
                current.add(pending);
            }
        }
        batches.add(current);
        log.info(String.format("%s batches created", batches.size()));
        return batches;
    }

    protected void LookupSubmissions() {
        String podId = statusProperties.getApplicationId();
        log.info("Getting all pending submission ids for podId {}", podId);
        Iterable<Pending> pendings = pendingRepo.findAllByPodId(podId);
        // we will either get the exact size or a 0 here.
        if (pendings.spliterator().getExactSizeIfKnown() == 0) {
            log.info("No pending submission ids for podId {}", podId);
            return;
        }
        // The MeF system has a limit of 100 submission ids checked per attempt
        List<Set<Pending>> batchedPendings = batchPendings(pendings);

        log.info("Creating ack client for podId {}", podId);
        long startTime = System.currentTimeMillis();
        Date date = new Date(startTime);
        log.info("Logging in at {} for podId {}", simpleDateFormat.format(date), podId);
        batchedPendings.forEach(batch -> {
            try {
                log.info("Getting acks from MeF for batch for podId {}", podId);
                getGetAcksResult(batch);
                // It might one day be necessary to purge submissionIds
                // It shouldn't be necessary with this service being only available
                // to our internal system.
            } catch (ToolkitException e) {
                // TODO: if a batch fails, find the bad one and report the problem to some other system
                log.error("Toolkit error getting ack on poll: {}", e.getMessage(), e);

                if (batch.size() == 1) {
                    createToolkitError(batch.iterator().next(), e);
                } else {
                    partitionBatch(batch);
                }
            } catch (ServiceException e) {
                log.error("Service error getting ack on poll: {}", e.getMessage(), e);
            }
        });

        long endTime = System.currentTimeMillis();
        date = new Date(endTime);
        log.info(
                "Logging out at {}, elapsed time in milliseconds: {}",
                simpleDateFormat.format(date),
                (endTime - startTime));
    }

    private void getGetAcksResult(Set<Pending> pendings) throws ToolkitException, ServiceException {
        Set<String> submissionIds =
                pendings.stream().map(Pending::getSubmissionId).collect(Collectors.toSet());
        GetAcksResultWrapper acknowledgements = getAcksClientService.getAcks(serviceContext, submissionIds);
        // to enable parallel processing of each batch of 100 submissions, we handle each batch of acknowledgements in a
        // separate thread
        // once we have fetched them from MeF
        new Thread(() -> bulkUpdateRecordsFromAckResultAndEnqueueStatusChangeMessages(acknowledgements, pendings))
                .start();
    }

    void bulkUpdateRecordsFromAckResultAndEnqueueStatusChangeMessages(
            GetAcksResultWrapper acksResult, Iterable<Pending> pendings) {
        /*
         * We start with a list of ackResults and the original Pending records in the database
         * All batching logic is handle as a map of a mefSubmissionId key to an arbitrary value
         * This allows us to make O(1) look ups between the various batching functions when creating new maps
         * There are several steps for map and transform steps before we are ready to save/write database records
         * and enqueue the SQS message back to the Backend service
         *
         * 1. createStatusSubmissionIdMap()
         *   - initializes a Map of ackStatus (accepted, rejected, pending) to an empty list
         *   - returns: {"accepted":[],"rejected":[],"pending":[]}
         * 2. Loop through the acks and:
         *   a) populate the submissionIdToValidationErrorMap map submission Id to a list of lists of strings representing each part of the validationErrorGroup
         *     - returns {"sub_id_1" :[["R0000-904-03","Reject and Stop","Software ID in the Return Header must have passed testing for the form family and ‘TaxYr’."],
         *               ["F1040-525-03","Reject and Stop","If 'PINTypeCd' in the Return Header has ..."]],"sub_id_2":[[...],[...]]}
         *   b) populate the statusSubmissionIdMap
         *       - returns: {"accepted":["sub_id_3"],"rejected":["sub_id_2",],"pending":["sub_id_1"]}
         *
         * 3. createNewCompleteds()
         *      - Prepare the new Completed entities *but do not save them yet*
         *
         * 4. Map the Completed and Pending entities to respective maps of submissionId:Completed and submissionId:Pending
         *
         * 5. bulkUpdateEntities()
         *      - bulkGetOrCreateErrorsToRejectedAcknowledgements()
         *           - Create the Error entities based on the submissionIdToValidationErrorMap and save them to the database
         *           - at this point, the Error entities are *not*  related to the Completed entities, because
         *           - Completed entities haven't been saved
         *           - returns submissionIdToError(), which is a map of submission Ids to Errors (newly created)
         *      - addErrorsToRejectedAcknowledgementsAndDeletePendingRecords()
         *           - pass the three maps (submissionIdToError,submissionIdToCompleted,submissionIdToPending) as args
         *           - iterate through the submissionIdToCompleted and:
         *               - if Errors exist, relate them to the completed
         *               - if the Pending exists, add it to the batch of Pendings to delete
         *               - in all cases, add the Completed to a batch of Completeds to persist
         *           - Save all Completeds in the batch to create
         *           - Delete all Pendings in the batch to delete
         * 6. Remove pending submissions from the map of statusSubmissionIds to produce finalStatusSubmissionIds
         * 7. Enqueue the finalStatusSubmissionIds to SQS and call statusChangeMessageService.publishStatusChangePayloadV1(finalStatusSubmissionIds)
         * */
        Map<String, List<String>> statusSubmissionIdMap = createStatusSubmissionIdMap();
        Map<String, List<List<String>>> submissionIdToValidationErrorMap = new HashMap<>();
        Map<String, Completed> submissionIdToCompletedMap = new HashMap<>();
        Map<String, Pending> submissionIdToPendingdMap = new HashMap<>();

        acksResult.getAcknowledgementsListWrapper().getAcknowledgements().forEach(acknowledgement -> {
            String status = acknowledgement.getAcceptanceStatusTxt().toLowerCase();
            String submissionId = acknowledgement.getSubmissionId();
            submissionIdToValidationErrorMap.put(submissionId, new ArrayList<>());
            if (status.equals("rejected")) {
                acknowledgement.getValidationErrorList().forEach(validationErrorGrp -> {
                    List<String> errorMap = new ArrayList<>();
                    errorMap.add(validationErrorGrp.getRuleNum());
                    errorMap.add(validationErrorGrp.getSeverityCd());
                    errorMap.add(validationErrorGrp.getErrorMessageTxt());
                    submissionIdToValidationErrorMap.get(submissionId).add(errorMap);
                });
            }
            addToSubmissionIdMap(status, submissionId, statusSubmissionIdMap);
        });

        Iterable<Completed> completeds = createNewCompleteds(statusSubmissionIdMap);
        completeds.forEach(completed -> submissionIdToCompletedMap.put(completed.getSubmissionId(), completed));
        pendings.forEach(pending -> submissionIdToPendingdMap.put(pending.getSubmissionId(), pending));

        bulkUpdateEntities(submissionIdToValidationErrorMap, submissionIdToCompletedMap, submissionIdToPendingdMap);
        Map<String, List<String>> finalStatusSubmissionIdMap =
                stripPendingAcknowledgementsFromStatusSubmissionIdMap(statusSubmissionIdMap);

        if (!finalStatusSubmissionIdMap.get("rejected").isEmpty()
                || !finalStatusSubmissionIdMap.get("accepted").isEmpty()) {
            statusChangeMessageService.publishStatusChangePayloadV1(finalStatusSubmissionIdMap);
        }
    }

    protected Map<String, List<String>> stripPendingAcknowledgementsFromStatusSubmissionIdMap(
            Map<String, List<String>> statusSubmssionIdMap) {
        statusSubmssionIdMap.remove("pending");
        return statusSubmssionIdMap;
    }

    protected Map<String, List<String>> createStatusSubmissionIdMap() {
        Map<String, List<String>> statusSubmissionIdMap = new HashMap<>();

        statusSubmissionIdMap.put("accepted", new ArrayList<>());
        statusSubmissionIdMap.put("rejected", new ArrayList<>());
        statusSubmissionIdMap.put("pending", new ArrayList<>());
        return statusSubmissionIdMap;
    }

    protected Iterable<Completed> createNewCompleteds(Map<String, List<String>> statusSubmissionIdMap) {
        List<Completed> completedsToCreate = new ArrayList<>();
        statusSubmissionIdMap.forEach((status, submissionIdSet) -> {
            if (status.equals("pending")) {
                log.info("The following subIds are still pending {}", submissionIdSet.toString());
            } else {
                submissionIdSet.forEach(submissionId -> {
                    Completed completed = new Completed();
                    completed.setSubmissionId(submissionId);
                    completed.setStatus(status);
                    completedsToCreate.add(completed);
                    log.info("{} acknowledgement received from MeF", status);
                });
            }
        });
        return completedsToCreate;
    }

    protected void addToSubmissionIdMap(
            String status, String submissionId, Map<String, List<String>> statusSubmissionIdMap) {
        MDC.put(AuditLogElement.mefSubmissionId.toString(), submissionId);
        if (statusSubmissionIdMap.containsKey(status)) {
            log.info(
                    "Acknowledgement with submissionId {} has status changed, adding to statusSubmissionIdMap",
                    submissionId);
            statusSubmissionIdMap.get(status).add(submissionId);
        } else {
            // default case where the status is not present or not one we expect
            // TODO: handle the exception status case when/if necessary
            log.error("Missing a status type: {}", status);
        }
        MDC.clear();
    }

    public void bulkUpdateEntities(
            Map<String, List<List<String>>> validationErrorMap,
            Map<String, Completed> submissionIdToCompleted,
            Map<String, Pending> submissionIdToPending) {
        Map<String, List<Error>> submissionIdToError =
                bulkGetOrCreateErrorsToRejectedAcknowledgements(validationErrorMap);
        addErrorsToRejectedAcknowledgementsAndDeletePendingRecords(
                submissionIdToError, submissionIdToCompleted, submissionIdToPending);
    }

    protected Map<String, List<Error>> bulkGetOrCreateErrorsToRejectedAcknowledgements(
            Map<String, List<List<String>>> validationErrorMap) {

        List<Error> errorList = new ArrayList<>();
        Map<String, List<Error>> submissionIdToError = new HashMap<>();

        validationErrorMap.forEach((submissionId, validationErrorGrp) -> {
            submissionIdToError.put(submissionId, new ArrayList<>());
            validationErrorGrp.forEach(validationError -> {
                String ruleNum = validationError.get(0);
                String severityCd = validationError.get(1);
                String errorMessageTxt = validationError.get(2);
                var databaseError = errorRepo.findById(ruleNum);
                if (databaseError.isPresent()) {
                    log.info(String.format("Found reject reason %s", ruleNum));
                    Error existingError = databaseError.get();
                    errorList.add(existingError);
                    submissionIdToError.get(submissionId).add(existingError);

                } else {
                    log.warn(String.format("New reject reason found: %s. New translation required", ruleNum));
                    Error newError = new Error();
                    newError.setMefErrorCode(ruleNum);
                    newError.setErrorCodeTranslationKey(CreateTranslationKey("reject", ruleNum));
                    newError.setMefErrorCategory(severityCd);
                    newError.setErrorMessage(errorMessageTxt);
                    log.info(String.format("Saving new reject reason %s", ruleNum));
                    errorList.add(newError);
                    submissionIdToError.get(submissionId).add(newError);
                }
            });
        });
        errorRepo.saveAll(errorList);
        return submissionIdToError;
    }

    protected void addErrorsToRejectedAcknowledgementsAndDeletePendingRecords(
            Map<String, List<Error>> submissionIdToError,
            Map<String, Completed> submissionIdToCompleted,
            Map<String, Pending> submissionIdToPending) {
        List<Completed> completedToSave = new ArrayList<>();
        List<Pending> pendingToDelete = new ArrayList<>();
        submissionIdToCompleted.forEach((submissionId, completed) -> {
            List<Error> errorsToPersist = submissionIdToError.get(submissionId);
            Optional<Pending> pending = Optional.ofNullable(submissionIdToPending.get(submissionId));
            if (errorsToPersist != null
                    && !errorsToPersist.isEmpty()
                    && completed.getStatus().toLowerCase().compareTo("rejected") == 0) {
                completed.setErrors(errorsToPersist);
            }
            completedToSave.add(completed);
            if (pending.isPresent()) {
                Pending p = pending.get();
                pendingToDelete.add(p);
            }
        });
        log.info(String.format("Creating %s completed objects", completedToSave.size()));
        completedRepo.saveAll(completedToSave);
        log.info(String.format("Deleting %s pending objects", pendingToDelete.size()));
        pendingRepo.deleteAll(pendingToDelete);
    }

    protected void createToolkitError(Pending pending, Exception e) {
        String submissionId = pending.getSubmissionId();
        ToolkitError tke = new ToolkitError();
        tke.setSubmissionId(submissionId);
        tke.setErrorName(e.getClass().getName());
        tke.setErrorMessage(e.toString());

        toolkitErrorRepo.save(tke);

        // delete the corresponding pending record
        try {
            deletePendingRecord(submissionId);
        } catch (EntityNotFoundException notFoundException) {
            log.error(notFoundException.getMessage());
        }
    }

    private void deletePendingRecord(String submissionId) {
        Pending p = pendingRepo
                .findById(submissionId)
                .orElseThrow(() -> new EntityNotFoundException(
                        String.format("Submission with id [%s] was not found in the database!", submissionId)));
        pendingRepo.delete(p);
    }

    private AcknowledgementStatus getCompletedStatus(Completed c) {
        if (c.getStatus().toLowerCase().compareTo("accepted") == 0) {
            return new AcknowledgementStatus(
                    Status.Accepted, CreateTranslationKey("status", "accepted"), List.of(), c.getCreatedAt());
        } else if (c.getStatus().toLowerCase().compareTo("rejected") == 0) {
            return new AcknowledgementStatus(
                    Status.Rejected,
                    CreateTranslationKey("status", "rejected"),
                    createRejectedReasonList(c),
                    c.getCreatedAt());
        } else {
            throw new RuntimeException("Missing status type");
        }
    }

    private List<RejectedStatus> createRejectedReasonList(Completed c) {
        List<RejectedStatus> statuses = new ArrayList<>();
        if (c.getErrors() != null) {
            c.getErrors()
                    .forEach(x -> statuses.add(new RejectedStatus(
                            x.getMefErrorCode(), x.getErrorCodeTranslationKey(), x.getErrorMessage())));
        }

        return statuses;
    }

    private String CreateTranslationKey(String type, String name) {
        return String.join(
                statusProperties.getTranslationKeySplitter(), statusProperties.getRootTranslationKey(), type, name);
    }

    private void partitionBatch(Set<Pending> pBatch) {
        Iterable<List<Pending>> partitions =
                Iterables.partition(pBatch, IntMath.divide(pBatch.size(), 2, RoundingMode.CEILING));

        partitions.forEach(partition -> {
            try {
                getGetAcksResult(new HashSet<>(partition));
            } catch (ToolkitException tke) {
                if (partition.size() == 1) {
                    // report and flag bad data to db
                    createToolkitError(partition.get(0), tke);
                } else {
                    partitionBatch(new HashSet<>(partition));
                }
            } catch (ServiceException se) {
                if (partition.size() == 1) {
                    log.warn("ServiceException - " + se);
                } else {
                    partitionBatch(new HashSet<>(partition));
                }
            }
        });
    }
}
