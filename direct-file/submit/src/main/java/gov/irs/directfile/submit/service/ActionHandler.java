package gov.irs.directfile.submit.service;

import java.text.SimpleDateFormat;
import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.message.SubmissionEventFailureCategoryEnum;
import gov.irs.directfile.models.message.SubmissionEventFailureDetailEnum;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;
import gov.irs.directfile.submit.actions.*;
import gov.irs.directfile.submit.actions.exception.BundleArchiveActionException;
import gov.irs.directfile.submit.actions.exception.CreateArchiveActionException;
import gov.irs.directfile.submit.actions.exception.SubmissionFailureException;
import gov.irs.directfile.submit.actions.results.BundleArchivesActionResult;
import gov.irs.directfile.submit.actions.results.CreateArchiveActionResult;
import gov.irs.directfile.submit.actions.results.SubmissionFailureActionResult;
import gov.irs.directfile.submit.command.*;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.SubmittedDataContainer;
import gov.irs.directfile.submit.domain.UserContextData;
import gov.irs.directfile.submit.exception.LoginFailureException;
import gov.irs.directfile.submit.exception.LogoutFailureException;
import gov.irs.directfile.submit.repository.PodIdentifierRepository;
import gov.irs.directfile.submit.service.interfaces.IBundleSubmissionActionHandler;
import gov.irs.directfile.submit.service.interfaces.ISubmissionFailureService;

@Slf4j
@Service
@SuppressWarnings({
    "PMD.ExceptionAsFlowControl",
    "PMD.ExcessiveParameterList",
    "PMD.SimpleDateFormatNeedsLocale",
    "PMD.UselessParentheses"
})
public class ActionHandler {
    private final SqsConnectionSetupService sqsConnectionSetupService;
    private final SubmissionConfirmationMessageService submissionConfirmationMessageService;
    protected final ActionContext context;
    private final ActionQueue actions;
    private final IBundleSubmissionActionHandler bundleSubmissionService;

    private final OfflineModeService offlineModeService;

    private final Set<SubmissionBatch> inProgressBatches;

    private SimpleDateFormat simpleDateFormat = new SimpleDateFormat("hh:mm:ss");
    private final BundleArchivesActionHandler bundleArchivesActionHandler;

    private final CleanupActionHandler cleanupActionHandler;
    private final CreateArchiveActionHandler createArchiveActionHandler;
    private final ISubmissionFailureService submissionFailureService;
    private final PodIdentifierRepository podIdentifierRepository;

    public ActionHandler(
            SqsConnectionSetupService sqsConnectionSetupService,
            SubmissionConfirmationMessageService submissionConfirmationMessageService,
            ActionQueue actions,
            ActionContext actionContext,
            IBundleSubmissionActionHandler bundleSubmissionService,
            OfflineModeService offlineModeService,
            Set<SubmissionBatch> inProgressBatches,
            BundleArchivesActionHandler bundleArchivesActionHandler,
            CleanupActionHandler cleanupActionHandler,
            CreateArchiveActionHandler createArchiveActionHandler,
            ISubmissionFailureService submissionFailureService,
            PodIdentifierRepository podIdentifierRepository) {
        this.sqsConnectionSetupService = sqsConnectionSetupService;
        this.submissionConfirmationMessageService = submissionConfirmationMessageService;
        this.context = actionContext;
        this.actions = actions;
        this.bundleSubmissionService = bundleSubmissionService;
        this.offlineModeService = offlineModeService;
        this.inProgressBatches = inProgressBatches;
        this.bundleArchivesActionHandler = bundleArchivesActionHandler;
        this.cleanupActionHandler = cleanupActionHandler;
        this.createArchiveActionHandler = createArchiveActionHandler;
        this.submissionFailureService = submissionFailureService;
        this.podIdentifierRepository = podIdentifierRepository;
    }

    @PostConstruct
    public void init() {
        updateAsid(context.getConfig());
    }

    protected void updateAsid(Config config) {
        String applicationId = config.getApplicationId();
        String asid = getAsid(config, applicationId);
        if (!Objects.equals(asid, config.getAsid()) && !asid.isEmpty() && !asid.isBlank()) {
            log.info(
                    "Setting new ASID ending in {} in Config and ServiceContext for {}",
                    asid.substring(asid.length() - 1),
                    applicationId);
            this.context.getConfig().setAsid(asid);
            this.context.getServiceContext().setAppSysID(asid);
        }
    }

    protected String getAsid(Config config, String applicationId) {
        log.info("Getting ASID for ActionContext update for {}", applicationId);
        String asid = config.getAsid();
        Optional<String> optAsid = podIdentifierRepository.findAsidByPodId(applicationId);
        if (optAsid.isPresent()) {
            log.info("Setting pod-specific ASID in Config for ActionContext for {}", applicationId);
            asid = optAsid.get();
        }
        return asid;
    }

    public void handleAction(Action action) {
        try {
            // These info logs don't need that info.
            log.info(String.format("running action: %s", action.getClass().getSimpleName()));
            switch (action.getType()) {
                case CREATE_ARCHIVE:
                    CreateArchiveActionResult archives =
                            createArchiveActionHandler.handleCommand((CreateArchiveAction) action);

                    if (!archives.getSubmissionArchiveContainers().isEmpty()) {
                        actions.getInProgressActions().add(new BundleArchiveAction(archives));
                    }
                    break;
                case BUNDLE_ARCHIVE:
                    log.info("Creating a bundle to send to the MeF");
                    BundleArchivesActionResult c =
                            bundleArchivesActionHandler.handleBundleCommand((BundleArchiveAction) action);
                    actions.getInProgressActions().add(new SubmitBundleAction(c));
                    break;
                case SUBMIT_BUNDLE:
                    log.info("Submit bundle action received");

                    SubmittedDataContainer submittedDataContainer;
                    if (context.getConfig().isSubmitActionEnabled() && !offlineModeService.getShouldStayOffline()) {
                        long startTime = System.currentTimeMillis();
                        Date date = new Date(startTime);
                        log.info("Logging in at {}", simpleDateFormat.format(date));
                        bundleSubmissionService.login();
                        log.info("Submitting bundle");

                        submittedDataContainer = bundleSubmissionService.handleCommand((SubmitBundleAction) action);
                        log.info(
                                "Successfully Submitted {} submissions for batch {} ",
                                submittedDataContainer.userContexts.size(),
                                submittedDataContainer.submissionBatch.path());
                        // TODO: call the ack system and let them know.
                        long endTime = System.currentTimeMillis();
                        date = new Date(endTime);
                        log.info(
                                "Logging out at {}, elapsed time in milliseconds: {}",
                                simpleDateFormat.format(date),
                                (endTime - startTime));
                        bundleSubmissionService.logout();
                        actions.getInProgressActions().add(new CleanupAction(submittedDataContainer.submissionBatch));

                        List<TaxReturnIdAndSubmissionId> taxReturnIdAndSubmissionIds =
                                submittedDataContainer.getTaxReturnIdAndSubmissionIds();
                        sqsConnectionSetupService.sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(
                                taxReturnIdAndSubmissionIds);

                        List<SubmissionConfirmationPayloadV2Entry> taxReturnSubmissionReceipts =
                                submittedDataContainer.getSuccessSubmissionConfirmationPayloadV2Entries();
                        submissionConfirmationMessageService.publishSubmissionConfirmationPayloadV2(
                                taxReturnSubmissionReceipts);
                    } else {
                        SubmitBundleAction bundleAction = (SubmitBundleAction) action;
                        log.info(
                                "Submit action not enabled. There were {} tax returns in batch {} ",
                                bundleAction
                                        .getBundleArchivesActionResult()
                                        .getBundledArchives()
                                        .UserContexts
                                        .size(),
                                bundleAction.getBundleArchivesActionResult().getBatch());

                        actions.getInProgressActions()
                                .add(new CleanupAction(bundleAction
                                        .getBundleArchivesActionResult()
                                        .getBatch()));
                    }
                    break;
                case SUBMISSION_FAILURE:
                    SubmissionFailureActionResult submissionFailureActionResult =
                            submissionFailureService.handleCommand((SubmissionFailureAction) action);
                    actions.getInProgressActions().add(new CleanupAction(submissionFailureActionResult.getBatch()));
                    break;
                case CLEANUP:
                    log.info("Cleaning up the file system");
                    cleanupActionHandler.handleAction((CleanupAction) action);
                    inProgressBatches.remove(((CleanupAction) action).getSubmissionBatch());
                    break;
                default:
                    throw new RuntimeException("missing type");
            }
        } catch (SubmissionFailureException e) {
            try {
                /*
                 * If Submission fails. Check if MeF is online by trying to logout.
                 * If logout succeeds (no exception is thrown), we know MeF is online, meaning there was an issue
                 * with the Submitted Bundle. Handle the Submission Failure by kicking off the retry process.
                 * */
                log.error(
                        "Failed to Submit Bundle for batch at path {} to MeF",
                        e.getBatch().path());
                bundleSubmissionService.logout();
                handleSubmissionFailure(e);
            } catch (LogoutFailureException logoutFailureException) {
                if (!offlineModeService.isOfflineModeEnabled()) {
                    log.info("Enabling Offline Mode. Unable to login.");
                    offlineModeService.enableOfflineMode();
                }
                // Re-Queue the failed action to be handled when we are back online
                actions.getInProgressActions().add(action);
            }

        } catch (LoginFailureException e) {
            try {
                log.error("Failed to log in to MeF");
                bundleSubmissionService.logout();
            } catch (LogoutFailureException logoutFailureException) {
                if (!offlineModeService.isOfflineModeEnabled()) {
                    log.info("Enabling Offline Mode. Unable to logout.");
                    offlineModeService.enableOfflineMode();
                }
                // Re-Queue the failed action to be handled when we are back online
                actions.getInProgressActions().add(action);
            }
        } catch (LogoutFailureException e) {
            if (!offlineModeService.isOfflineModeEnabled()) {
                log.info("Enabling Offline Mode. Unable to logout.");
                offlineModeService.enableOfflineMode();
            }
            // Re-Queue the failed action to be handled when we are back online
            actions.getInProgressActions().add(action);
        } catch (BundleArchiveActionException e) {
            handleBundleArchiveActionException(e);
        } catch (CreateArchiveActionException e) {
            log.error("CreateArchiveActionException", e);
        } catch (ActionException e) {
            log.error("action exception", e);
        } catch (JsonProcessingException e) {
            log.error("JsonProcessingException", e);
        } catch (Exception e) {
            log.error("Unknown error caught in Action Handler", e);
        }
    }

    private void handleBundleArchiveActionException(BundleArchiveActionException e) {
        log.error(
                "BundleArchiveActionException occurred. Unable to bundle the following tax returns: {}",
                e.userContextDataTaxReturnIdsToString(),
                e);

        // publish failure message containing taxreturn submission in the batch
        List<UserContextData> userContextDataList = e.getUserContextDataList();
        List<SubmissionConfirmationPayloadV2Entry> entries = userContextDataList.stream()
                .map(userContextData -> {
                    // Create failureEventMetadata
                    Map<String, String> failureEventMetadata = new HashMap<>();
                    failureEventMetadata.put(
                            "errorMessage", String.format("BundleArchiveAction failed, %s", e.getMessage()));
                    failureEventMetadata.put(
                            "failureCategory", SubmissionEventFailureCategoryEnum.PROCESSING.getFailureCategory());
                    failureEventMetadata.put(
                            "failureDetail", SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING.getFailureDetail());

                    // Create message payload for Submission Confirmation queue
                    return getFailureEventSubmissionConfirmationPayloadV2Entry(userContextData, failureEventMetadata);
                })
                .toList();
        submissionConfirmationMessageService.publishSubmissionConfirmationPayloadV2(entries);
        actions.getInProgressActions().add(new CleanupAction(e.getBatch()));
    }

    private void handleSubmissionFailure(SubmissionFailureException e) {
        int numberOfSubmissions = e.getBundledArchives().UserContexts.size();
        if (numberOfSubmissions > 1) {
            actions.getInProgressActions().add(new SubmissionFailureAction(e));
        } else if (numberOfSubmissions == 1) {
            UserContextData failedSubmissionUserContext =
                    e.getBundledArchives().UserContexts.get(0);
            log.error(String.format(
                    "Failed to submit return to MeF for user %s", failedSubmissionUserContext.getUserId()));

            Map<String, String> failureEventMetadata = new HashMap<>();
            // TODO: These should probably be keys somewhere
            failureEventMetadata.put("errorMessage", String.format("Submission to MeF failed, %s", e.getMessage()));
            failureEventMetadata.put(
                    "failureCategory", SubmissionEventFailureCategoryEnum.PROCESSING.getFailureCategory());
            failureEventMetadata.put(
                    "failureDetail", SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING.getFailureDetail());

            SubmissionConfirmationPayloadV2Entry entry = getFailureEventSubmissionConfirmationPayloadV2Entry(
                    failedSubmissionUserContext, failureEventMetadata);
            submissionConfirmationMessageService.publishSubmissionConfirmationPayloadV2(List.of(entry));
            actions.getInProgressActions().add(new CleanupAction(e.getBatch()));
        } else {
            log.error("Unable to submit bundle to MeF because batch is empty.");
        }
    }

    private static SubmissionConfirmationPayloadV2Entry getFailureEventSubmissionConfirmationPayloadV2Entry(
            UserContextData failedSubmissionUserContext, Map<String, String> failureEventMetadata) {
        SubmissionConfirmationPayloadV2Entry entry = new SubmissionConfirmationPayloadV2Entry(
                new TaxReturnSubmissionReceipt(
                        UUID.fromString(failedSubmissionUserContext.getTaxReturnId()),
                        failedSubmissionUserContext.getSubmissionId(),
                        null,
                        null),
                SubmissionEventTypeEnum.FAILED,
                failureEventMetadata);
        return entry;
    }
}
