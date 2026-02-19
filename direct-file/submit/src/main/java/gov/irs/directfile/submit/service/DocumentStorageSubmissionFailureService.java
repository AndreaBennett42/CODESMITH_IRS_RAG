package gov.irs.directfile.submit.service;

import java.time.Clock;
import java.time.Year;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.actions.results.SubmissionFailureActionResult;
import gov.irs.directfile.submit.command.SubmissionFailureAction;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.domain.DocumentStoreResource;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.service.interfaces.ISubmissionFailureService;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@Slf4j
@Service
@SuppressFBWarnings(
        value = {"NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
public class DocumentStorageSubmissionFailureService implements ISubmissionFailureService {
    private static final String S3_SUBMISSIONS_FOLDER = "pre-submission-batching";

    private final ISynchronousDocumentStoreService synchronousDocumentStoreService;
    private final String applicationId;
    private final Clock clock;

    public DocumentStorageSubmissionFailureService(
            ISynchronousDocumentStoreService synchronousDocumentStoreService,
            @Value("${submit.application-id}") String applicationId,
            Clock clock) {
        this.synchronousDocumentStoreService = synchronousDocumentStoreService;
        this.applicationId = applicationId;
        this.clock = clock;
    }

    @Override
    public void Setup(Config config) throws Throwable {}

    /**
     * This method is responsible for copying files of a failed batch (Unable to Submit to MeF)
     * to an error path that will re-submit each indiviudal submission to MeF as it's own batch.
     *
     * If a batch contains 10 submissions. This method will copy the files over
     * such that we will re-submit 10 batches, each of size 1, when our asynchronous ErrorBatchPoller picks them up.
     * */
    @Override
    public void processFailedBatch(SubmissionBatch batch) {
        /**
         *
         * A batch stores submissions at this path.
         * BATCH_PATH: pre-submission-batching/{configurable-application-id}/{batch-control-year}/{batch-number}/
         *
         * The files for a given submission are  at:
         *
         * BATCH_PATH/{submissionId}/manifest.xml
         *   => pre-submission-batching/{configurable-application-id}/{batch-control-year}/{batch-number}/{submissionId}/manifest.xml
         * BATCH_PATH/{submissionId}/return.xml
         *  => pre-submission-batching/{configurable-application-id}/{batch-control-year}/{batch-number}/{submissionId}/return.xml
         * BATCH_PATH/{submissionId}/userContext.json
         *  => pre-submission-batching/{configurable-application-id}/{batch-control-year}/{batch-number}/{submissionId}/userContext.json
         *
         * This method copies each of the files of a submission into its own folder that will
         * be treated as a batch.
         *
         * An error batch stores the files for a submission that will be treated as its own batch:
         * ERROR_PATH:  pre-submission-batching/{configurable-application-id}/{tax-year}/errors/{batch-number}/
         *
         * To treat a submission as its own batch, we will copy files to a unique path for each submission.
         * We assign a submission number between 0 and the size of the batch.
         * INDIVIDUAL_ERROR_BATCH = ERROR_PATH/{submissionNumber}
         *
         * Files for the submission in this batch will live at:
         *
         * INDIVIDUAL_ERROR_BATCH/manifest.xml
         *   => pre-submission-batching/{configurable-application-id}/{tax-year}/errors/{batch-number}/{submissionNumber}/{submissionId}/manifest.xml
         * INDIVIDUAL_ERROR_BATCH/return.xml
         *  =>  pre-submission-batching/{configurable-application-id}/{tax-year}/errors/{batch-number}/{submissionNumber}/{submissionId}/return.xml
         * INDIVIDUAL_ERROR_BATCH/userContext.json
         *  => pre-submission-batching/{configurable-application-id}/{tax-year}/errors/{batch-number}/{submissionNumber}/{submissionId}/manifest.xml
         *
         * */
        log.info("Processing Failed Batch for path: " + batch.path());
        String errorPath = String.format(
                "%s/errors/%s/%s/%s", S3_SUBMISSIONS_FOLDER, applicationId, getBatchControlYear(), batch.batchId());
        // 1. Get the objects in the batch
        List<DocumentStoreResource> objects = synchronousDocumentStoreService.getObjectKeys(batch.path());

        // 2. For each key, copy to the error path
        List<String> submissionIdObjectKeys = synchronousDocumentStoreService.getSubFolders(batch.path());
        Map<String, Integer> submissionIdToSubmissionNumber = new HashMap<>();

        /**
         * We're going to treat each submission as its own batch.
         * We are assigning a submissionNumber to each submission in the batch.
         *
         * */
        for (int i = 0; i < submissionIdObjectKeys.size(); i++) {
            submissionIdToSubmissionNumber.put(submissionIdObjectKeys.get(i), i);
        }

        for (DocumentStoreResource documentStoreResource : objects) {
            String fullPath = documentStoreResource.getFullLocation();

            // 2a. Find the path to files of a submission. This looks like
            // "pre-submission-batching/{application-id}/{batch-control-year}/{batchId}/{submissionId}"
            String submissionIdObjectKey =
                    fullPath.substring(0, fullPath.indexOf('/', batch.path().length()) + 1);

            // 2b. Find the filename associated with the submission. For example, this will look like
            // {submissionId}/manifest.xml
            String fileName = fullPath.substring(batch.path().length());
            int submissionNumber = submissionIdToSubmissionNumber.get(submissionIdObjectKey);
            // 2c. Generate the path where the error batch will live

            // pre-submission-batching/errors/{applicationId}/{batchControlYear}/{batchId}/{submissionNumber}/{submissionId}/file.{xml,json}
            String individualBatchErrorKey = String.format("%s/%s/%s", errorPath, submissionNumber, fileName);
            synchronousDocumentStoreService.copyObject(documentStoreResource, individualBatchErrorKey);
        }
        log.info("Successfully moved failed batch to error bucket:" + batch.path());
    }

    @Override
    public SubmissionFailureActionResult handleCommand(SubmissionFailureAction submissionFailureActionCommand) {
        this.processFailedBatch(
                submissionFailureActionCommand.getSubmissionFailureException().getBatch());
        return new SubmissionFailureActionResult(
                submissionFailureActionCommand.getSubmissionFailureException().getBatch());
    }

    private int getBatchControlYear() {
        return Year.now(clock).getValue() - 1;
    }
}
