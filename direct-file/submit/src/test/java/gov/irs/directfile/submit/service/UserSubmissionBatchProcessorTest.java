package gov.irs.directfile.submit.service;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.submit.command.ActionType;
import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.repository.interfaces.IBatchRepository;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class UserSubmissionBatchProcessorTest {
    private static final String APPLICATION_ID = "application-id";
    ActionQueue actionQueue;
    Set<SubmissionBatch> inProgressBatches;
    UserSubmissionBatchProcessor subject;

    IBatchRepository batchRepository = mock(IBatchRepository.class);

    OfflineModeService offlineModeService = new OfflineModeService();

    @BeforeEach
    public void setup() {
        actionQueue = new ActionQueue();
        inProgressBatches = new HashSet<>();
        subject = new UserSubmissionBatchProcessor(
                actionQueue, batchRepository, APPLICATION_ID, offlineModeService, inProgressBatches);
    }

    @Test
    public void itPublishesSubmissionToActionQueue() {
        SubmissionBatch batch = new SubmissionBatch(0L, "path/to/files");

        // Act: call process batch
        subject.processBatch(batch);

        // Assert: Expect a Create Archive Action was added to the queue
        assertEquals(1, actionQueue.getNewActions().size());
        assertEquals(1, inProgressBatches.size());
    }

    @Test
    public void itPublishesOldBatchesThatExistToActionQueue() throws InterruptedException {
        // Arrange:
        when(batchRepository.getCurrentWritingBatch(eq(APPLICATION_ID))).thenReturn(3L);

        List<SubmissionBatch> submissionBatches =
                List.of(new SubmissionBatch(1L, "path1"), new SubmissionBatch(2L, "path2"));
        when(batchRepository.getUnprocessedBatches(eq(APPLICATION_ID))).thenReturn(submissionBatches);

        // Act:
        subject.processOldBatches();

        // Assert: Expect two CREATE_ARCHIVE actions were added to the queue
        assertEquals(2, actionQueue.getNewActions().size());
        assertEquals(
                ActionType.CREATE_ARCHIVE, actionQueue.getNewActions().take().getType());
        assertEquals(
                ActionType.CREATE_ARCHIVE, actionQueue.getNewActions().take().getType());
    }

    @Test
    public void itDoesNotPublishOldBatchesThatAreAlreadyBeingProcessedToActionQueue() {
        // Arrange: Add a batch to the inProgressBatches object
        SubmissionBatch inProgressBatch = new SubmissionBatch(1L, "/batchId/1");
        inProgressBatches.add(inProgressBatch);

        List<SubmissionBatch> submissionBatches = List.of(inProgressBatch);
        when(batchRepository.getUnprocessedBatches(eq(APPLICATION_ID))).thenReturn(submissionBatches);

        // Act: Call process old batches
        subject.processOldBatches();

        // Assert: Expect the action queue to be empty because we do not add batches that are already being processed.
        assertEquals(0, actionQueue.getNewActions().size());
    }

    @Test
    public void itDoesNotProcessBatchesWHenOfflineModeEnabled() {
        // Arrange: Enable offline mode
        offlineModeService.enableOfflineMode();
        SubmissionBatch batch = new SubmissionBatch(0L, "path/to/files");

        // Act: call process batch
        subject.processBatch(batch);

        // Assert: Expect the action queue is empty because we do not publish actions when offline mode is enabled
        assertTrue(actionQueue.getNewActions().isEmpty());
    }
}
