package gov.irs.directfile.submit.service;

import java.time.Clock;
import java.time.Year;
import java.util.List;
import java.util.Set;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.actions.ActionContext;
import gov.irs.directfile.submit.command.CreateArchiveAction;
import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.storagelocations.StorageLocationBuilder;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@Service
@Slf4j
public class ErrorBatchPoller {

    private final ActionQueue actions;
    private final Set<SubmissionBatch> inProgressBatches;

    private final ISynchronousDocumentStoreService storageService;
    private final String applicationId;

    private final ActionContext context;
    private final Clock clock;

    public ErrorBatchPoller(
            ActionQueue actions,
            ISynchronousDocumentStoreService storageService,
            @Value("${submit.application-id}") String applicationId,
            Set<SubmissionBatch> inProgressBatches,
            ActionContext context,
            Clock clock) {
        this.actions = actions;
        this.storageService = storageService;
        this.applicationId = applicationId;
        this.inProgressBatches = inProgressBatches;
        this.context = context;
        this.clock = clock;
    }

    @Scheduled(fixedRateString = "${submit.batching.errorPollingMilliseconds}", initialDelay = 1000L)
    public void poll() {
        if (!context.getConfig().isSubmitActionEnabled()) {
            log.info("Submit action is disabled, not polling for error batches");
            return;
        }
        String errorPath = StorageLocationBuilder.getErrorFolderLocation(applicationId, getBatchControlYear());
        List<String> errorSubfolders = storageService.getSubFolders(errorPath);
        log.info("Polling for error batches...");
        if (!errorSubfolders.isEmpty()) {
            log.info(String.format(
                    "Found %s error batches in poller. Re-processing as individual submissions",
                    errorSubfolders.size()));
            /*
             * errorSubfolders is a list of all the object paths for failed batches in the error subfolder.
             * Each entry in this list is the path to a batch that failed, it has been copied to the error path.
             * Each entry looks like this:
             * pre-submission-batching/errors/{applicationId}/{batchControlYear}/{batch-id}/
             *
             * */
            for (String errorSubFolder : errorSubfolders) {
                /*
                 * errorSubFolderBatches is a list paths to each submission that will be retried as an individual batch.
                 * Each string in this list is formatted like this:
                 * pre-submission-batching/{configurable-application-id}/{tax-year}/errors/{batch-number}/{submissionNumber}
                 *
                 * This path allows us to treat each submission as its own batch.
                 *
                 */
                List<String> errorSubFolderBatches = storageService.getSubFolders(errorSubFolder);

                for (String errorBatchPath : errorSubFolderBatches) {

                    /*
                    errorBatchPath looks like this:
                    pre-submission-batching/{configurable-application-id}/{batchControlYear}/errors/{batch-number}/{submissionNumber}

                    Get the submissionNumber and treat is as the batch id for this batch.

                    */
                    String[] split = errorBatchPath.split("/");
                    long submissionNumber = Long.parseLong(split[split.length - 1]);
                    SubmissionBatch submissionBatch = new SubmissionBatch(submissionNumber, errorBatchPath);

                    if (!inProgressBatches.contains(submissionBatch)) {
                        actions.getNewActions().add(new CreateArchiveAction(submissionBatch));
                        inProgressBatches.add(submissionBatch);
                    }
                }
            }
        }
    }

    private int getBatchControlYear() {
        return Year.now(clock).getValue() - 1;
    }
}
