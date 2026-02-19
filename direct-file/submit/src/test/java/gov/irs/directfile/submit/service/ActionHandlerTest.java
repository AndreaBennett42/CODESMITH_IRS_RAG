package gov.irs.directfile.submit.service;

import java.util.*;

import lombok.SneakyThrows;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import gov.irs.a2a.mef.mefheader.TestCdType;
import gov.irs.mef.exception.ToolkitException;
import gov.irs.mef.inputcomposition.PostmarkedSubmissionArchive;
import gov.irs.mef.inputcomposition.SubmissionBuilder;
import gov.irs.mef.services.ServiceContext;

import gov.irs.directfile.audit.events.TinType;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.submit.Runner;
import gov.irs.directfile.submit.actions.ActionContext;
import gov.irs.directfile.submit.actions.BundleArchivesActionHandler;
import gov.irs.directfile.submit.actions.CleanupActionHandler;
import gov.irs.directfile.submit.actions.CreateArchiveActionHandler;
import gov.irs.directfile.submit.actions.exception.SubmissionFailureException;
import gov.irs.directfile.submit.actions.results.BundleArchivesActionResult;
import gov.irs.directfile.submit.actions.results.CreateArchiveActionResult;
import gov.irs.directfile.submit.command.ActionType;
import gov.irs.directfile.submit.command.BundleArchiveAction;
import gov.irs.directfile.submit.command.SubmitBundleAction;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.DirectoriesConfig;
import gov.irs.directfile.submit.domain.*;
import gov.irs.directfile.submit.domain.model.PodIdentifier;
import gov.irs.directfile.submit.exception.LoginFailureException;
import gov.irs.directfile.submit.exception.LogoutFailureException;
import gov.irs.directfile.submit.mocks.FakeSynchronousDocumentStorageService;
import gov.irs.directfile.submit.mocks.MutableTestClock;
import gov.irs.directfile.submit.repository.PodIdentifierRepository;
import gov.irs.directfile.submit.service.interfaces.IBundleSubmissionActionHandler;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@DataJpaTest
@ExtendWith(MockitoExtension.class)
public class ActionHandlerTest {

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

    private Runner runner;
    private String softwareId = "12345678";
    private String softwareVersionNum = "2023.0.1";

    private static final String applicationId = "ApplicationId";

    private ActionHandler actionHandler;
    private ActionQueue actions = new ActionQueue();
    private Set<SubmissionBatch> inProgressBatches = new HashSet<>();
    private final Set<SubmissionBatch> inProgressSubmissions = new HashSet<>();

    @Mock
    private SqsConnectionSetupService sqsConnectionSetupService;

    @Mock
    private SubmissionConfirmationMessageService submissionConfirmationMessageService;

    @Mock
    private IBundleSubmissionActionHandler bundleSubmissionService;

    @Autowired
    PodIdentifierRepository podIdentifierRepository;

    private OfflineModeService offlineModeService;

    private ISynchronousDocumentStoreService synchronousDocumentStoreService;

    @BeforeEach
    public void setup() {
        synchronousDocumentStoreService = new FakeSynchronousDocumentStorageService();
        ActionContext actionContext = new ActionContext(createConfig());
        offlineModeService = new OfflineModeService();
        actionHandler = new ActionHandler(
                sqsConnectionSetupService,
                submissionConfirmationMessageService,
                actions,
                actionContext,
                bundleSubmissionService,
                offlineModeService,
                inProgressSubmissions,
                new BundleArchivesActionHandler(actionContext),
                new CleanupActionHandler(actionContext, synchronousDocumentStoreService),
                new CreateArchiveActionHandler(actionContext, synchronousDocumentStoreService),
                new DocumentStorageSubmissionFailureService(
                        synchronousDocumentStoreService, applicationId, new MutableTestClock()),
                podIdentifierRepository);
        runner = new Runner(actions, actionHandler, offlineModeService);
    }

    @Test
    public void itAddsSubmissionFailureActionToQueueWhenFailedSubmissionBatchHasMultipleSubmissions() throws Exception {

        // Arrange: Create a SubmitBundleAction,
        SubmissionBatch batch = new SubmissionBatch(0L, "env/batches/0");
        UserContextData userContextData = new UserContextData(
                "submissionId",
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");

        UserContextData userContextData2 = new UserContextData(
                "submissionId",
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextData, userContextData2), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleAction = new SubmitBundleAction(bundleArchivesActionResult);

        when(bundleSubmissionService.handleCommand(any()))
                .thenThrow(new SubmissionFailureException(batch, bundledArchives, new RuntimeException()));
        // Act: Call handleAction(submitBundleAction)
        actionHandler.handleAction(submitBundleAction);

        // Assert: Expect a SubmissionFailureAction was added to the queue
        assertEquals(1, actions.getInProgressActions().size());
        assertEquals(
                ActionType.SUBMISSION_FAILURE,
                actions.getInProgressActions().peek().getType());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, never()).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    public void itDoesNotRetryBatchWhenOnlySubmissionInABatch() throws Exception {
        // Arrange: Create a SubmitBundleAction, only add one submission for the batch
        SubmissionBatch batch = new SubmissionBatch(0L, "env/batches/0");
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "submissionId";
        UserContextData userContextData = new UserContextData(
                submissionId, "userId", taxReturnId.toString(), "userTin", TinType.INDIVIDUAL, "0.0.0", "2024-01-01");
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextData), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleAction = new SubmitBundleAction(bundleArchivesActionResult);
        when(bundleSubmissionService.handleCommand(any()))
                .thenThrow(new SubmissionFailureException(batch, bundledArchives, new RuntimeException()));

        // Act: Call handleAction(submitBundleAction)
        actionHandler.handleAction(submitBundleAction);
        // A cleanup action is added to the queue after submit bundle. Call handleAction() to cleanup the batch
        actionHandler.handleAction(actions.getInProgressActions().take());

        // Assert: Expect a SubmissionFailureAction was not added to the queue
        assertEquals(0, actions.getInProgressActions().size());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, times(1)).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    public void itEnablesOfflineModeWhenLoginFails() throws Exception {
        // Arrange: Create a SubmitBundleAction, only add one submission for the batch
        SubmissionBatch batch = new SubmissionBatch(0L, "env/batches/0");
        UserContextData userContextData = new UserContextData(
                "submissionId",
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextData), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleAction = new SubmitBundleAction(bundleArchivesActionResult);
        when(bundleSubmissionService.login()).thenThrow(new LoginFailureException("Login Failure"));
        when(bundleSubmissionService.logout()).thenThrow(new LogoutFailureException("Logout Failure"));
        // Act: Call handleAction(submitBundleAction)
        actionHandler.handleAction(submitBundleAction);

        // Assert: Expect a OfflineMode to be enabled due to login failure
        assertTrue(offlineModeService.isOfflineModeEnabled());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, never()).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    public void itEnablesOfflineModeWhenLogoutFails() throws Exception {
        // Arrange: Create a SubmitBundleAction, only add one submission for the batch
        SubmissionBatch batch = new SubmissionBatch(0L, "env/batches/0");
        UserContextData userContextData = new UserContextData(
                "submissionId",
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextData), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleActionCommand = new SubmitBundleAction(bundleArchivesActionResult);
        when(bundleSubmissionService.logout()).thenThrow(new LogoutFailureException("Logout Failure"));
        SendSubmissionsResultWrapper sendSubmissionsResultWrapper = null;
        when(bundleSubmissionService.handleCommand(any()))
                .thenReturn(
                        new SubmittedDataContainer(bundledArchives.UserContexts, sendSubmissionsResultWrapper, batch));
        // Act: Call handleAction(submitBundleAction)
        actionHandler.handleAction(submitBundleActionCommand);

        // Assert: Expect a OfflineMode to be enabled due to login failure
        assertTrue(offlineModeService.isOfflineModeEnabled());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, never()).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    public void itEnablesOfflineModeWhenSubmissionFailsAndMefIsOffline() throws Exception {
        // Arrange: Create a SubmitBundleAction, only add one submission for the batch
        SubmissionBatch batch = new SubmissionBatch(0L, "env/batches/0");
        UserContextData userContextData = new UserContextData(
                "submissionId",
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextData), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleActionCommand = new SubmitBundleAction(bundleArchivesActionResult);
        // Throw Submission Failure Exception when we call submitBundle()
        when(bundleSubmissionService.handleCommand(any()))
                .thenThrow(new SubmissionFailureException(
                        batch, bundledArchives, new RuntimeException("Submission Failed")));
        when(bundleSubmissionService.logout()).thenThrow(LogoutFailureException.class);

        // Act:
        actionHandler.handleAction(submitBundleActionCommand);

        // Assert: Expect a OfflineMode to be enabled due to login failure
        assertTrue(offlineModeService.isOfflineModeEnabled());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, never()).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    public void itRetriesFailedSubmissionWhenSubmissionFailsAndMefIsOnline() throws Exception {
        // Arrange: Create a SubmitBundleAction with two submissions
        SubmissionBatch batch = new SubmissionBatch(0L, "env/batches/0");
        UserContextData userContextDataA = new UserContextData(
                "submissionId",
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");
        UserContextData userContextDataB = new UserContextData(
                "B-submissionId",
                "B-userId",
                UUID.randomUUID().toString(),
                "B-userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextDataA, userContextDataB), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleActionCommand = new SubmitBundleAction(bundleArchivesActionResult);

        // Throw Submission Failure Exception when we call submit bundles
        when(bundleSubmissionService.handleCommand(any()))
                .thenThrow(new SubmissionFailureException(
                        batch, bundledArchives, new RuntimeException("Submission Failed")));
        // Return true when attempting to log out. Indicating Mef is online
        when(bundleSubmissionService.login()).thenReturn(true);
        when(bundleSubmissionService.logout()).thenReturn(true);

        // Act:
        actionHandler.handleAction(submitBundleActionCommand);

        // Assert: Expect a OfflineMode is not enable because MeF is online
        assertFalse(offlineModeService.isOfflineModeEnabled());

        // Expect a SubmissionFailureAction was added to the queue to process the failed submissions
        assertEquals(
                ActionType.SUBMISSION_FAILURE,
                actions.getInProgressActions().peek().getType());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, never()).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    public void itSendsFailureMessageToSubmissionConfirmationQueueContaingInfoForAllTaxReturnsInTheBatch()
            throws Exception {
        // Arrange: Create a SubmitBundleAction with two submissions
        final UserContextData userContextData1 = new UserContextData(
                "00000",
                "00000000-0000-0000-0000-000000000000",
                "11111111-1111-1111-1111-111111111111",
                "111001111",
                TinType.INDIVIDUAL,
                "0.0.0.0",
                "2024-01-01");
        final UserContextData userContextData2 = new UserContextData(
                "11111",
                "88888888-8888-8888-8888-888888888888",
                "99999999-9999-9999-9999-999999999999",
                "111002222",
                TinType.INDIVIDUAL,
                "1.1.1.1",
                "2024-01-01");
        final PostmarkedSubmissionArchive mockSubmissionArchive1 = mock(PostmarkedSubmissionArchive.class);
        final PostmarkedSubmissionArchive mockSubmissionArchive2 = mock(PostmarkedSubmissionArchive.class);

        final List<SubmissionArchiveContainer> submissionArchiveContainers = List.of(
                new SubmissionArchiveContainer(userContextData1, mockSubmissionArchive1),
                new SubmissionArchiveContainer(userContextData2, mockSubmissionArchive2));

        CreateArchiveActionResult createArchiveActionResult =
                new CreateArchiveActionResult(new SubmissionBatch(0L, ""), submissionArchiveContainers);

        BundleArchiveAction bundleArchiveActionCommand = new BundleArchiveAction(createArchiveActionResult);

        try (MockedStatic<SubmissionBuilder> mockSubmissionBuilder = Mockito.mockStatic(SubmissionBuilder.class)) {
            mockSubmissionBuilder
                    .when(() -> SubmissionBuilder.createSubmissionContainer(
                            any(PostmarkedSubmissionArchive[].class), anyString()))
                    .thenThrow(new ToolkitException("test"));

            // Act:
            actionHandler.handleAction(bundleArchiveActionCommand);

            // Assert
            ArgumentCaptor<List<SubmissionConfirmationPayloadV2Entry>> argumentCaptor =
                    ArgumentCaptor.forClass(List.class);

            // Verify calls to message queues/topics
            verify(sqsConnectionSetupService, never())
                    .sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
            verify(submissionConfirmationMessageService, times(1))
                    .publishSubmissionConfirmationPayloadV2(argumentCaptor.capture());

            assertEquals(
                    userContextData1.getTaxReturnId(),
                    argumentCaptor
                            .getValue()
                            .get(0)
                            .getTaxReturnSubmissionReceipt()
                            .getTaxReturnId()
                            .toString());
            assertEquals(
                    userContextData1.getSubmissionId(),
                    argumentCaptor
                            .getValue()
                            .get(0)
                            .getTaxReturnSubmissionReceipt()
                            .getSubmissionId());

            assertEquals(
                    userContextData2.getTaxReturnId(),
                    argumentCaptor
                            .getValue()
                            .get(1)
                            .getTaxReturnSubmissionReceipt()
                            .getTaxReturnId()
                            .toString());
            assertEquals(
                    userContextData2.getSubmissionId(),
                    argumentCaptor
                            .getValue()
                            .get(1)
                            .getTaxReturnSubmissionReceipt()
                            .getSubmissionId());
        }
    }

    @Test
    @SneakyThrows
    public void itCleansUpFileSystemForSubmissionFailureOfSingleSubmission() {
        // Arrange: Create a batch, user object, and add manifest.xml, userContext.json, submission.xml to document
        // storage
        long batchId = 0L;
        Long batchSubmissionNumber = 5L;
        String submissionId = "submissionId";
        String errorBatchPath = String.format(
                "pre-submission-batching/errors/dfsys-mef-submit-deployment-0-us-gov-east-1/2023/%s/%s/%s",
                batchId, batchSubmissionNumber, submissionId);
        SubmissionBatch batch = new SubmissionBatch(batchId, errorBatchPath);
        UserContextData userContextDataA = new UserContextData(
                submissionId,
                "userId",
                UUID.randomUUID().toString(),
                "userTin",
                TinType.INDIVIDUAL,
                "0.0.0",
                "2024-01-01");

        String manifestXmlKey = errorBatchPath + "/manifest.xml";
        String userContextJsonKey = errorBatchPath + "/userContext.json";
        String submissionXmlKey = errorBatchPath + "/submission.xml";
        synchronousDocumentStoreService.write(manifestXmlKey, "<xml>My Tax Return Manifest </xml>");
        synchronousDocumentStoreService.write(userContextJsonKey, "{}");
        synchronousDocumentStoreService.write(submissionXmlKey, "<xml>My Tax Return Submission </xml>");
        // Expect document storage to have 3 files at the errorBatchPath. The 3 files we just added via
        // synchronousDocumentStoreService.write()
        assertEquals(
                3, synchronousDocumentStoreService.getObjectKeys(errorBatchPath).size());

        // Arrange Part 2: Set up the bundleSubmissionService to throw an exception
        BundledArchives bundledArchives = new BundledArchives(List.of(userContextDataA), null);
        BundleArchivesActionResult bundleArchivesActionResult =
                new BundleArchivesActionResult(new SubmissionBatch(0L, ""), bundledArchives);

        SubmitBundleAction submitBundleActionCommand = new SubmitBundleAction(bundleArchivesActionResult);

        // Throw Submission Failure Exception when we call submit bundles
        when(bundleSubmissionService.handleCommand(any()))
                .thenThrow(new SubmissionFailureException(
                        batch, bundledArchives, new RuntimeException("Submission Failed")));
        // Return true when attempting to log out. Indicating Mef is online
        when(bundleSubmissionService.login()).thenReturn(true);
        when(bundleSubmissionService.logout()).thenReturn(true);

        // Act: Call SubmitBundle via the ActionHandler and throw error when calling SubmitBundler
        actionHandler.handleAction(submitBundleActionCommand);
        // Call handleAction again to run the cleanup action that's in the queue
        actionHandler.handleAction(actions.getInProgressActions().take());
        // Assert: Expect document storage is empty for files at the path of the error batch, because we clean up
        // filesystem for failed submissions
        assertEquals(
                0, synchronousDocumentStoreService.getObjectKeys(errorBatchPath).size());

        // Verify calls to message queues/topics
        verify(sqsConnectionSetupService, never()).sendListOfSubmissionAndTaxReturnIdsToPendingSubmissionQueue(any());
        verify(submissionConfirmationMessageService, times(1)).publishSubmissionConfirmationPayloadV2(any());
    }

    @Test
    void whenUpdateAsid_WithPodIdentifiers_ThenReturnsAsidFromDatabase_ThenCreatesCorrectServiceContext() {
        Config config = createConfig();
        PodIdentifier p = createPodIdentifer("us-gov-east-1", config.getAsid(), 0);
        podIdentifierRepository.save(p);

        String asid = podIdentifierRepository.findAsidByPodId(p.getPodId()).get();

        actionHandler.updateAsid(config);
        ServiceContext serviceContext = actionHandler.context.getServiceContext();
        Config updatedConfig = actionHandler.context.getConfig();
        assertNotNull(serviceContext);
        assertNotNull(updatedConfig);
        assertEquals(serviceContext.getAppSysID(), config.getAsid());
        assertEquals(serviceContext.getAppSysID(), asid);
        assertEquals(serviceContext.getEtin().toString(), config.getEtin());
        assertEquals(serviceContext.getTestCdType(), TestCdType.T);

        assertEquals(updatedConfig.getAsid(), asid);
    }

    @Test
    void whenUpdateAsid_WithNoPodIdentifiers_ThenReturnsDefaultAsid_ThenCreatesCorrectServiceContext() {
        Config config = createConfig();

        Optional<String> nonExistentAsid = podIdentifierRepository.findAsidByPodId(config.getApplicationId());
        assertTrue(nonExistentAsid.isEmpty());

        actionHandler.updateAsid(config);
        ServiceContext serviceContext = actionHandler.context.getServiceContext();
        Config updatedConfig = actionHandler.context.getConfig();
        assertNotNull(serviceContext);
        assertNotNull(updatedConfig);
        assertEquals(serviceContext.getAppSysID(), config.getAsid());
        assertEquals(serviceContext.getEtin().toString(), config.getEtin());
        assertEquals(serviceContext.getTestCdType(), TestCdType.T);
    }

    @Test
    void whenUpdateAsid_WithMultiplePodIdentifiers_ThenReturnsCorrectAsid_ThenCreatesCorrectServiceContext() {
        Config config = createConfig();
        String ASID = "asid1235";

        PodIdentifier p1 = createPodIdentifer("us-gov-east-1", ASID, 2);
        PodIdentifier p2 = createPodIdentifer("us-gov-east-1", ASID + "2", 1);
        PodIdentifier p3 = createPodIdentifer("us-gov-east-1", config.getAsid(), 0);
        podIdentifierRepository.save(p1);
        podIdentifierRepository.save(p2);
        podIdentifierRepository.save(p3);

        String correctAsid =
                podIdentifierRepository.findAsidByPodId(p3.getPodId()).get();
        actionHandler.updateAsid(config);
        ServiceContext serviceContext = actionHandler.context.getServiceContext();
        Config updatedConfig = actionHandler.context.getConfig();

        assertNotNull(serviceContext);
        assertNotNull(updatedConfig);
        assertEquals(serviceContext.getAppSysID(), config.getAsid());
        assertEquals(serviceContext.getAppSysID(), correctAsid);
        assertEquals(serviceContext.getEtin().toString(), config.getEtin());
        assertEquals(serviceContext.getTestCdType(), TestCdType.T);

        assertEquals(updatedConfig.getAsid(), correctAsid);
    }

    private PodIdentifier createPodIdentifer(String region, String asid, int index) {
        PodIdentifier p = new PodIdentifier();
        p.setAsid(asid);
        p.setRegion(region);
        p.setPodId("dfsys-mef-status-deployment-" + String.valueOf(index) + "-" + region);
        return p;
    }

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
                softwareId,
                softwareVersionNum,
                "dfsys-mef-submit-deployment-0-us-gov-east-1");
    }
}
