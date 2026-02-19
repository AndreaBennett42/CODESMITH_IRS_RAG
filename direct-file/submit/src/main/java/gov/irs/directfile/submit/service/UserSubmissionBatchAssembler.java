package gov.irs.directfile.submit.service;

import java.time.Clock;
import java.time.Year;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.BatchingProperties;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.UserSubmission;
import gov.irs.directfile.submit.repository.DocumentStorageBatchRepository;
import gov.irs.directfile.submit.repository.interfaces.IBatchRepository;

@Service
@EnableScheduling
@SuppressWarnings("PMD.SignatureDeclareThrowsException")
public class UserSubmissionBatchAssembler {
    private final IBatchRepository batchRepository;

    private final BatchingProperties batchingProperties;
    private final String applicationId;
    private final UserSubmissionBatchProcessor userSubmissionBatchProcessor;

    // Since the onTimeout() scheduled task runs in a separate thread from the rest
    // of this service, we need to ensure atomic updates to the batch to avoid losing
    // any submissions due to race conditions.
    private final AtomicLong currentBatchId = new AtomicLong(0);
    private final AtomicInteger currentBatchSize = new AtomicInteger(0);
    private final Clock clock;

    public UserSubmissionBatchAssembler(
            IBatchRepository batchRepository,
            BatchingProperties batchingProperties,
            @Value("${submit.application-id}") String applicationId,
            UserSubmissionBatchProcessor userSubmissionBatchProcessor,
            Clock clock) {
        this.batchRepository = batchRepository;
        this.batchingProperties = batchingProperties;
        this.applicationId = applicationId;
        this.userSubmissionBatchProcessor = userSubmissionBatchProcessor;
        this.clock = clock;
    }

    @PostConstruct
    public void setup() throws Exception {
        currentBatchId.set(batchRepository.getCurrentWritingBatch(applicationId));
    }

    @Scheduled(fixedRateString = "${submit.batching.batchTimeoutMilliseconds}")
    public void onTimeout() {
        synchronized (this) {
            submitBatchForProcessing(currentBatchId.get());
        }
    }

    private void submitBatchForProcessing(long batchId) {
        if (currentBatchSize.get() > 0) {
            userSubmissionBatchProcessor.processBatch(new SubmissionBatch(
                    batchId,
                    DocumentStorageBatchRepository.generateLocationForBatch(
                            applicationId, batchId, getBatchControlYear())));
            currentBatchId.incrementAndGet();
        }
        currentBatchSize.set(0);
    }

    public void addSubmission(UserSubmission userSubmission) throws Exception {
        synchronized (this) {
            long currentBatch = currentBatchId.get();
            String batchPath = DocumentStorageBatchRepository.generateLocationForBatch(
                    applicationId, currentBatch, getBatchControlYear());
            batchRepository.addSubmission(new SubmissionBatch(currentBatch, batchPath), userSubmission);

            if (currentBatchSize.incrementAndGet() >= batchingProperties.getMaxBatchSize()) {
                submitBatchForProcessing(currentBatchId.get());
            }
        }
    }

    private int getBatchControlYear() {
        return Year.now(clock).getValue() - 1;
    }
}
