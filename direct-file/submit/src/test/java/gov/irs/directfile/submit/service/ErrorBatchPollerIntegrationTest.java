package gov.irs.directfile.submit.service;

import java.time.Clock;
import java.time.Year;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.submit.actions.ActionContext;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.DirectoriesConfig;
import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.mocks.FakeSynchronousDocumentStorageService;
import gov.irs.directfile.submit.mocks.MutableTestClock;

@ExtendWith(MockitoExtension.class)
public class ErrorBatchPollerIntegrationTest {
    private static final String APPLICATION_ID = "APPLICATION_ID";
    private final ActionQueue actions = new ActionQueue();
    private final Set<SubmissionBatch> inProgresssBatches = new HashSet<>();

    private final ActionContext context = new ActionContext(createConfig());
    private static final Clock sharedClock = new MutableTestClock();

    DocumentStorageSubmissionFailureService submissionFailureService;
    ErrorBatchPoller errorBatchPoller;
    FakeSynchronousDocumentStorageService fakeSynchronousDocumentStorageService =
            new FakeSynchronousDocumentStorageService();

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
        submissionFailureService = new DocumentStorageSubmissionFailureService(
                fakeSynchronousDocumentStorageService, APPLICATION_ID, sharedClock);
        errorBatchPoller = new ErrorBatchPoller(
                actions,
                fakeSynchronousDocumentStorageService,
                APPLICATION_ID,
                inProgresssBatches,
                context,
                sharedClock);
    }

    @AfterEach
    public void teardown() {
        actions.getInProgressActions().clear();
        actions.getNewActions().clear();
        inProgresssBatches.clear();
        fakeSynchronousDocumentStorageService.clear();
    }

    @Test
    public void itPublishesErrorBatchesToActionQueue() {
        // 1. Add a batch with some submissions to the fake synchronous store
        int batchControlYear = Year.now(sharedClock).getValue() - 1;
        long batchId = 10L;

        List<String> submissionIds = List.of("subId-ABC", "subId-DEF", "subId-GHI");
        String submissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, batchId);

        for (String submissionId : submissionIds) {
            addFakeSubmission(submissionBatchObjectKey, submissionId);
        }

        SubmissionBatch batch = new SubmissionBatch(batchId, submissionBatchObjectKey);

        // 2. Call the process error batch, to create an inditivual batch for each submission
        submissionFailureService.processFailedBatch(batch);

        // 3. Because we'll treat each submission as an individual batch, we expect to find 3 subfolders for the path.
        String errorPath = String.format(
                "pre-submission-batching/errors/%s/%s/%s/", APPLICATION_ID, batchControlYear, batch.batchId());
        List<String> errorFolders = fakeSynchronousDocumentStorageService.getSubFolders(errorPath);

        Assertions.assertEquals(3, errorFolders.size());

        // 4. Call ErrorBatchPoller.poll()
        errorBatchPoller.poll();

        // 5. Expect 3 actions were put onto the queue, one for each submission in the batch
        Assertions.assertEquals(3, actions.getNewActions().size());
        Assertions.assertEquals(3, inProgresssBatches.size());
    }

    @Test
    public void itDoesNotPublishInProgressErrorBatchesToTheQueue() {
        // 1. Add a batch with some submissions to the fake synchronous store
        int batchControlYear = Year.now(sharedClock).getValue() - 1;
        long batchId = 10L;

        List<String> submissionIds = List.of("subId-ABC", "subId-DEF", "subId-GHI");
        String submissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, batchId);

        for (String submissionId : submissionIds) {
            addFakeSubmission(submissionBatchObjectKey, submissionId);
        }

        SubmissionBatch batch = new SubmissionBatch(batchId, submissionBatchObjectKey);

        // 2. Call the process error batch, to create an inditivual batch for each submission
        submissionFailureService.processFailedBatch(batch);

        // 3. Because we'll treat each submission as an individual batch, we expect to find 3 subfolders for the path.
        String errorPath = String.format(
                "pre-submission-batching/errors/%s/%s/%s/", APPLICATION_ID, batchControlYear, batch.batchId());
        List<String> errorFolders = fakeSynchronousDocumentStorageService.getSubFolders(errorPath);

        Assertions.assertEquals(3, errorFolders.size());

        // 4. Call ErrorBatchPoller.poll() twice.
        errorBatchPoller.poll();
        errorBatchPoller.poll();

        // 5. Expect 3 actions were put onto the queue, even though we called poll twice because they were already in
        // progress
        Assertions.assertEquals(3, actions.getNewActions().size());
        Assertions.assertEquals(3, inProgresssBatches.size());
    }

    private void addFakeSubmission(String batchObjectKey, String submissionId) {
        fakeSynchronousDocumentStorageService.write(
                batchObjectKey + submissionId + "/submission.xml", "submission.xml");
        fakeSynchronousDocumentStorageService.write(batchObjectKey + submissionId + "/manifest.xml", "manifest.xml");
        fakeSynchronousDocumentStorageService.write(
                batchObjectKey + submissionId + "/userContext.json", "userContext.json");
    }
}
