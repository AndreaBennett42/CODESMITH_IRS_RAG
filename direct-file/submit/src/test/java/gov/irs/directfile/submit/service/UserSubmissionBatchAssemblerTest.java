package gov.irs.directfile.submit.service;

import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.submit.BatchingProperties;
import gov.irs.directfile.submit.domain.UserSubmission;
import gov.irs.directfile.submit.mocks.MutableTestClock;
import gov.irs.directfile.submit.repository.interfaces.IBatchRepository;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class UserSubmissionBatchAssemblerTest {
    @Mock
    private IBatchRepository batchRepository;

    @Mock
    private UserSubmissionBatchProcessor batchProcessor;

    private UserSubmissionBatchAssembler batchAssembler;

    @BeforeEach
    public void setup() {
        BatchingProperties batchProperties = new BatchingProperties(3, 100);
        batchAssembler = new UserSubmissionBatchAssembler(
                batchRepository,
                batchProperties,
                "dfsys-mef-submit-deployment-0-us-gov-east-1",
                batchProcessor,
                new MutableTestClock());
    }

    @Test
    public void onTimeout_submitsEmptyBatch() {
        batchAssembler.onTimeout();

        verify(batchProcessor, times(0)).processBatch(any());
    }

    @Test
    public void onTimeout_submitsNonEmptyBatch() throws Exception {
        addSubmission();
        batchAssembler.onTimeout();

        verify(batchProcessor, times(1)).processBatch(any());
    }

    @Test
    public void addSubmission_doesNotSubmitBatch() throws Exception {
        // Batch size is 3, so just add 1 submission.  Should not process batch.
        addSubmission();

        verify(batchProcessor, times(0)).processBatch(any());
    }

    @Test
    public void addSubmission_submitsBatch() throws Exception {
        // Batch size is 3, so add 3 submissions.  Should process batch.
        addSubmission();
        addSubmission();
        addSubmission();

        verify(batchProcessor, times(1)).processBatch(any());
    }

    private void addSubmission() throws Exception {
        batchAssembler.addSubmission(new UserSubmission(
                UUID.randomUUID().toString(),
                UUID.randomUUID().toString(),
                "123456789",
                "/path/to/manifest",
                "/path/to/submission",
                "/path/to/context"));
    }
}
