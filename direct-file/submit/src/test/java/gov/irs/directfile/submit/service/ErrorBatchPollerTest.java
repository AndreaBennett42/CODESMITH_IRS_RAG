package gov.irs.directfile.submit.service;

import java.time.Year;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.submit.actions.ActionContext;
import gov.irs.directfile.submit.command.ActionType;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.DirectoriesConfig;
import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.storagelocations.StorageLocationBuilder;
import gov.irs.directfile.submit.mocks.FakeSynchronousDocumentStorageService;
import gov.irs.directfile.submit.mocks.MutableTestClock;

@ExtendWith(MockitoExtension.class)
public class ErrorBatchPollerTest {
    private static final String APPLICATION_ID = "application-id";
    private MutableTestClock testClock = new MutableTestClock();

    private final ActionQueue actions = new ActionQueue();
    private final Set<SubmissionBatch> inProgresssBatches = new HashSet<>();
    private final FakeSynchronousDocumentStorageService fakeDocumentStoreService =
            new FakeSynchronousDocumentStorageService();

    private ErrorBatchPoller errorBatchPoller;

    ActionContext context = new ActionContext(createConfig());

    private Config createConfig() {
        boolean runnerDisabledForTesting = false;
        DirectoriesConfig directoriesConfig = new DirectoriesConfig(
                "src/test/resources/test",
                "src/test/resources/test",
                "src/test/resources/test",
                "src/test/resources/test",
                "src/test/resources/test",
                "src/test/resources/test");
        return new Config(
                "Test",
                null,
                null,
                directoriesConfig,
                null,
                null,
                "12345",
                "12345",
                "12345",
                "",
                runnerDisabledForTesting,
                true,
                true,
                "12345",
                "",
                "",
                "dfsys-mef-submit-deployment-0-us-gov-east-1");
    }

    // NOTE: Needed because we're mocking MEF Classes. MeF SDK expects this env variable A2A_TOOLKIT_HOME to be defined
    @BeforeAll
    public static void setupSystemProperties() {
        String userDirectory = System.getProperty("user.dir");
        System.setProperty("A2A_TOOLKIT_HOME", userDirectory + "/src/test/resources/");
    }

    @AfterAll
    public static void cleanupSystemProperties() {
        System.clearProperty("A2A_TOOLKIT_HOME");
    }

    @BeforeEach
    public void setup() {
        errorBatchPoller = new ErrorBatchPoller(
                actions, fakeDocumentStoreService, APPLICATION_ID, inProgresssBatches, context, testClock);
    }

    @AfterEach
    public void teardown() {
        actions.getNewActions().clear();
        actions.getInProgressActions().clear();
        inProgresssBatches.clear();
        fakeDocumentStoreService.clear();
    }

    @Test
    public void itReprocessesEachSubmissionOfAFailedBatchAsAsItsOwnBatch() {
        // Arrange: Add 2 submissions to the error path
        SubmissionBatch batch = new SubmissionBatch(2L, String.format("/path/%s", 2L));
        createFakeErrorBatchForSubmission(batch, "ABC-123", 0); // Assume there is a 0-th submission for the batch
        createFakeErrorBatchForSubmission(batch, "XYZ-789", 1); // Assume there is a 1-th submission for the batch

        // Act: Call Poll
        errorBatchPoller.poll();

        // Assert: Expect two create archive actions on the queue - one for each submission in the original batch
        Assertions.assertEquals(2, actions.getNewActions().size());
        Assertions.assertEquals(2, inProgresssBatches.size());
    }

    @Test
    public void itDoesNotReprocessSubmissionsThatAreAlreadyInProgress() {
        // Arrange: Add 2 submissions to the error path. Add a SubmissionBatch to inProgressBatches to represent it
        // being in progress
        int batchControlYear = Year.now(testClock).getValue() - 1;
        SubmissionBatch batch = new SubmissionBatch(2L, String.format("/path/%s", 2L));
        createFakeErrorBatchForSubmission(batch, "ABC-123", 0); // Assume there is a 0-th submission for the batch
        createFakeErrorBatchForSubmission(batch, "XYZ-789", 1); // Assume there is a 1-th submission for the batch
        String path = String.format(
                "%s%s/%s/",
                StorageLocationBuilder.getErrorFolderLocation(APPLICATION_ID, batchControlYear),
                batch.batchId(),
                0L // Creating a batch for the 0-th submission
                );
        inProgresssBatches.add(new SubmissionBatch(0L, path));

        // Act: Call Poll
        errorBatchPoller.poll();

        // Assert: Expect only 1 action was added to the queue, because the 0-th submission was in progress
        Assertions.assertEquals(1, actions.getNewActions().size());
        Assertions.assertEquals(
                ActionType.CREATE_ARCHIVE,
                actions.getNewActions().stream().toList().get(0).getType());
    }

    private void createFakeErrorBatchForSubmission(
            SubmissionBatch submissionBatch, String submissionId, int submissionNumber) {
        int batchControlYear = Year.now(testClock).getValue() - 1;
        String errorPath = StorageLocationBuilder.getErrorFolderLocation(APPLICATION_ID, batchControlYear);
        String errorBatchPath = errorPath + submissionBatch.batchId() + "/" + submissionNumber + "/" + submissionId;

        fakeDocumentStoreService.write(errorBatchPath + "/" + "submission.xml", "submission xml");
        fakeDocumentStoreService.write(errorBatchPath + "/" + "manifest.xml", "manifest xml");
        fakeDocumentStoreService.write(errorBatchPath + "/" + "userContext.json", "user context json");
    }
}
