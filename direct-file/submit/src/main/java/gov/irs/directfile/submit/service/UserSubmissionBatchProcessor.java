package gov.irs.directfile.submit.service;

import java.util.List;
import java.util.Set;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.command.CreateArchiveAction;
import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.repository.interfaces.IBatchRepository;

@Service
@Slf4j
public class UserSubmissionBatchProcessor {
    private final ActionQueue actions;
    private final IBatchRepository batchRepository;
    private final String applicationId;

    private final OfflineModeService offlineModeService;

    private final Set<SubmissionBatch> inProgressBatches;

    @Autowired
    public UserSubmissionBatchProcessor(
            ActionQueue actions,
            IBatchRepository batchRepository,
            @Value("${submit.application-id}") String applicationId,
            OfflineModeService offlineModeService,
            Set<SubmissionBatch> inProgressBatches) {
        this.actions = actions;
        this.batchRepository = batchRepository;
        this.applicationId = applicationId;
        this.offlineModeService = offlineModeService;
        this.inProgressBatches = inProgressBatches;
    }

    public void processOldBatches() {
        if (!offlineModeService.isOfflineModeEnabled()) {
            // 1. Get the current batch
            // 2. From n - 1 to 0 AND exists(path) processBatch

            List<SubmissionBatch> submissionBatches = batchRepository.getUnprocessedBatches(applicationId);

            submissionBatches.forEach(batch -> {
                if (!inProgressBatches.contains(batch)) {
                    log.info("Processing Old batch: " + batch.batchId() + " " + "for path " + batch.path());
                    this.processBatch(batch);
                    inProgressBatches.add(batch);
                } else {
                    log.info("Batch for path " + batch.path() + " is already in progress.");
                }
            });

        } else {
            log.info("Submit App is in offline mode. No batches will be processed at this time.");
        }
    }

    public void processBatch(SubmissionBatch submissionBatch) {
        if (offlineModeService.isOfflineModeEnabled()) {
            log.info(
                    "UserSubmissionBatchProcessor: Submit App is in offline mode. No batches will be processed until offline mode is disabled.");
        } else {
            if (!inProgressBatches.contains(submissionBatch)) {
                CreateArchiveAction createArchiveActionCommand = new CreateArchiveAction(submissionBatch);
                log.info("Adding batch" + submissionBatch + "to action queue");
                actions.getNewActions().add(createArchiveActionCommand);
                inProgressBatches.add(submissionBatch);
            } else {
                log.info("Batch for path " + submissionBatch.path() + " is already in progress.");
            }
        }
    }
}
