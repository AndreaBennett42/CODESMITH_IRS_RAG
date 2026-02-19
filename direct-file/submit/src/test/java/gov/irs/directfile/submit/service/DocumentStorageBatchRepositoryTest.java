package gov.irs.directfile.submit.service;

import java.time.*;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.mef.inputcomposition.BinaryAttachment;
import gov.irs.mef.inputcomposition.SubmissionManifest;
import gov.irs.mef.inputcomposition.SubmissionXML;

import gov.irs.directfile.audit.events.TinType;
import gov.irs.directfile.submit.BatchingProperties;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.SubmissionData;
import gov.irs.directfile.submit.domain.UserContextData;
import gov.irs.directfile.submit.domain.UserSubmission;
import gov.irs.directfile.submit.mocks.FakeSynchronousDocumentStorageService;
import gov.irs.directfile.submit.mocks.MutableTestClock;
import gov.irs.directfile.submit.repository.DocumentStorageBatchRepository;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class DocumentStorageBatchRepositoryTest {
    private static final int MAX_BATCH_SIZE = 5;
    private static final long BATCH_TIMEOUT_MILLISECONDS = 10;

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

    private static final String APPLICATION_ID = "application-id-123";
    private DocumentStorageBatchRepository subject;
    private BatchingProperties batchingProperties = BatchingProperties.builder()
            .maxBatchSize(MAX_BATCH_SIZE)
            .batchTimeoutMilliseconds(10_000) // 10 seconds
            .build();

    private MutableTestClock testClock;

    private FakeSynchronousDocumentStorageService documentStorageService;

    @BeforeEach
    public void setup() {
        // 1. Initialize services that need to be registered by the services container
        var date = LocalDateTime.of(2050, 11, 5, 14, 1);

        var instant = date.toInstant(ZoneOffset.UTC);
        testClock = new MutableTestClock(instant, ZoneId.of("UTC"));
        documentStorageService = new FakeSynchronousDocumentStorageService(testClock);
        subject = new DocumentStorageBatchRepository(
                documentStorageService, APPLICATION_ID, batchingProperties, testClock);
    }

    @AfterEach
    public void teardown() {
        documentStorageService.clear();
    }

    @Test
    public void itReturnsZeroBatchIdWhenNoBatchesExist() {
        // Arrange: Do nothing - no batches written

        // Act: Call get current writing batch
        Long submissionBatchId = subject.getCurrentWritingBatch(APPLICATION_ID);

        // Assert: Expect the batch id to be 0
        assertEquals(0L, submissionBatchId);

        // Path /pre-submission-batching/taxYear/batchId
        int bactchControlYear = getBatchControlYear();
        String expectedPath = String.format(
                "pre-submission-batching/%s/%s/%s/", APPLICATION_ID, bactchControlYear, submissionBatchId);

        assertEquals(
                expectedPath,
                DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, submissionBatchId, getBatchControlYear()));
    }

    @Test
    public void itReturnsHighestBatchPlusOneWhenCurrentBatchTimeoutHasExpired() throws Exception {
        // Arrange: Add data to document store for a batch
        long previousBatchId = 100L;
        int batchControlYear = getBatchControlYear();
        String submissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, previousBatchId);
        SubmissionBatch b = new SubmissionBatch(previousBatchId, submissionBatchObjectKey);

        SubmissionData fakeSubmissionData = mockSubmissionData();

        // 2/6/2024: replaced the call to generate with pre-written data
        documentStorageService.write("/manifest", fakeSubmissionData.Manifest.getXmlData());
        documentStorageService.write("/submission", fakeSubmissionData.Xml.getXmlData());
        documentStorageService.write("/userContext", fakeSubmissionData.UserContext.toString());

        subject.addSubmission(
                b, new UserSubmission("userId", "taxReturnId", "123456", "/manifest", "/submission", "/userContext"));

        // Act: Update the clock to move forward by batchTimeout milliseconds and 1 second. Call get
        // current batch
        testClock.fastForward(Duration.of(batchingProperties.getBatchTimeoutMilliseconds() + 1000, ChronoUnit.MILLIS));
        Long currentBatchId = subject.getCurrentWritingBatch(APPLICATION_ID);

        // Assert: Expect the current batch will be 1 + current highest batch number
        assertEquals(previousBatchId + 1, currentBatchId);
        String expectedPath =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, currentBatchId);
        assertEquals(
                expectedPath,
                DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, currentBatchId, getBatchControlYear()));
    }

    @Test
    public void itReturnsCurrentBatchWhenTimeoutHasNotExpired() throws Exception {
        // Arrange: Add data to document store for a batch
        long previousBatchId = 100L;
        int batchControlYear = getBatchControlYear();
        String originalSubmissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, previousBatchId);
        SubmissionBatch b = new SubmissionBatch(previousBatchId, originalSubmissionBatchObjectKey);

        SubmissionData fakeSubmissionData = mockSubmissionData();

        // 2/6/2024: replaced the call to generate with pre-written data
        documentStorageService.write("/manifest", fakeSubmissionData.Manifest.getXmlData());
        documentStorageService.write("/submission", fakeSubmissionData.Xml.getXmlData());
        documentStorageService.write("/userContext", fakeSubmissionData.UserContext.toString());

        subject.addSubmission(
                b, new UserSubmission("userId", "taxReturnId", "123456", "/manifest", "/submission", "/userContext"));

        // Act: Update the clock to move forward by batchTimeout milliseconds minus 1 second. Call get
        // current batch
        testClock.fastForward(Duration.of(batchingProperties.getBatchTimeoutMilliseconds() - 1000, ChronoUnit.MILLIS));
        Long currentBatch = subject.getCurrentWritingBatch(APPLICATION_ID);

        // Assert: Expect the current batch will be previousBatchID because it's still within the same time window
        assertEquals(previousBatchId, currentBatch.longValue());
        assertEquals(
                originalSubmissionBatchObjectKey,
                DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, currentBatch, getBatchControlYear()));
    }

    @Test
    public void itReturnsCurrentBatchWhenTimeoutIsReached() throws Exception {
        // Arrange: Add data to document store for a batch
        long previousBatchId = 100L;
        int batchControlYear = getBatchControlYear();
        String originalSubmissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, previousBatchId);
        SubmissionBatch b = new SubmissionBatch(previousBatchId, originalSubmissionBatchObjectKey);

        SubmissionData fakeSubmissionData = mockSubmissionData();

        // 2/6/2024: replaced the call to generate with pre-written data
        documentStorageService.write("/manifest", fakeSubmissionData.Manifest.getXmlData());
        documentStorageService.write("/submission", fakeSubmissionData.Xml.getXmlData());
        documentStorageService.write("/userContext", fakeSubmissionData.UserContext.toString());

        subject.addSubmission(
                b, new UserSubmission("userId", "taxReturnId", "123456", "/manifest", "/submission", "/userContext"));
        // Act: Update the clock to move forward by exactly batchTimeout milliseconds minus one millisecond. Call get
        // current batch
        testClock.fastForward(Duration.of(batchingProperties.getBatchTimeoutMilliseconds(), ChronoUnit.MILLIS));
        Long currentBatch = subject.getCurrentWritingBatch(APPLICATION_ID);

        // Assert: Expect the current batch will be previousBatchID because it's still within the same time window
        assertEquals(previousBatchId, currentBatch);
        assertEquals(
                originalSubmissionBatchObjectKey,
                DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, currentBatch, getBatchControlYear()));
    }

    @Test
    public void itReturnsEmptyOptionalWhenNoBatchFoundForBatchId() {
        // Arrange: N/A

        // Act: Try to retrieve a batch, when no batches exist in the repository
        Optional<SubmissionBatch> submissionBatchOptional = subject.getSubmissionBatch(APPLICATION_ID, 100L);
        assertTrue(submissionBatchOptional.isEmpty());
    }

    @Test
    public void itReturnsSubmissionBatchIfOneExists() throws Exception {
        // Arrange: Add data to document store for a batch
        long batchId = 100L;
        int batchControlYear = getBatchControlYear();
        String originalSubmissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, batchId);
        SubmissionBatch b = new SubmissionBatch(batchId, originalSubmissionBatchObjectKey);
        SubmissionData fakeSubmissionData = mockSubmissionData();

        // 2/6/2024: replaced the call to generate with pre-written data
        documentStorageService.write("/manifest", fakeSubmissionData.Manifest.getXmlData());
        documentStorageService.write("/submission", fakeSubmissionData.Xml.getXmlData());
        documentStorageService.write("/userContext", fakeSubmissionData.UserContext.toString());

        subject.addSubmission(
                b, new UserSubmission("userId", "taxReturnId", "123456", "/manifest", "/submission", "/userContext"));
        // Act: Call getSubmissionBatch
        Optional<SubmissionBatch> submissionBatchOptional = subject.getSubmissionBatch(APPLICATION_ID, batchId);

        assertTrue(submissionBatchOptional.isPresent());
        SubmissionBatch batch = submissionBatchOptional.get();
        assertEquals(batchId, batch.batchId());
        assertEquals(originalSubmissionBatchObjectKey, batch.path());
    }

    @Test
    public void itReturnsCurrentBatchIdPlusOneWhenMaxBatchSizeIsReached() throws Exception {
        // Arrange: Create a batch, and add MAX_BATCH_SIZE submissions to the batch
        Instant now = Instant.now();
        testClock.set(now);

        long batchId = 100L;
        int batchControlYear = getBatchControlYear();
        String originalSubmissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, batchId);
        SubmissionBatch b = new SubmissionBatch(batchId, originalSubmissionBatchObjectKey);

        for (int i = 0; i < MAX_BATCH_SIZE; i++) {
            String submissionId = UUID.randomUUID().toString();
            SubmissionData fakeSubmissionData = mockSubmissionData(submissionId);

            // 2/6/2024: replaced the call to generate with pre-written data
            documentStorageService.write("/manifest", fakeSubmissionData.Manifest.getXmlData());
            documentStorageService.write("/submission", fakeSubmissionData.Xml.getXmlData());
            documentStorageService.write("/userContext", fakeSubmissionData.UserContext.toString());

            subject.addSubmission(
                    b,
                    new UserSubmission(
                            "userId", "taxReturnId", submissionId, "/manifest", "/submission", "/userContext"));
        }

        // Act: Call getCurrentWritingBatch
        Long submissionBatch = subject.getCurrentWritingBatch(APPLICATION_ID);
        // Expect the batch id of the current writing batch is batchId + 1 because there are more than MAX_BATCH_SiZE
        // submissions in the current batch
        assertEquals(batchId + 1, submissionBatch);
    }

    private SubmissionData mockSubmissionData() {
        return mockSubmissionData("submissionId");
    }

    private SubmissionData mockSubmissionData(String submissionId) {
        String userId = "UserId";
        String taxReturnId = "taxReturnId";
        String userTin = "userTin";
        TinType userTinType = TinType.INDIVIDUAL;
        String remoteAddress = "0.0.0.0";
        String signDate = "2024-01-01";
        UserContextData userContextData =
                new UserContextData(submissionId, userId, taxReturnId, userTin, userTinType, remoteAddress, signDate);

        int taxYear = getBatchControlYear();
        String manifestXmlString = "<hello></hello>" + "<TaxYr>" + taxYear + "</TaxYr>";
        SubmissionManifest submissionManifest = mock(SubmissionManifest.class);
        when(submissionManifest.getXmlData()).thenReturn(manifestXmlString);

        String submissionXMLString = "<hello>" + "<TaxYr>" + taxYear + "</TaxYr>" + "</hello>";
        SubmissionXML submissionXML = mock(SubmissionXML.class);
        when(submissionXML.getXmlData()).thenReturn(submissionXMLString);
        BinaryAttachment[] binaryAttachments = new BinaryAttachment[] {};
        return new SubmissionData(userContextData, submissionManifest, submissionXML, binaryAttachments);
    }

    @Test
    public void getUnprocessedBatches_filtersOutCurrentBatchFromResult() throws Exception {
        long currentBatch = 101L;
        long previousBatch = 100L;
        int batchControlYear = getBatchControlYear();
        String currentSubmissionBatchKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, currentBatch);
        String previousSubmissionBatchKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, previousBatch);
        SubmissionBatch currentSubmissionBatch = new SubmissionBatch(currentBatch, currentSubmissionBatchKey);
        SubmissionBatch previousSubmissionBatch = new SubmissionBatch(previousBatch, previousSubmissionBatchKey);

        SubmissionData fakeSubmissionData = mockSubmissionData();

        // 2/6/2024: replaced the call to generate with pre-written data
        documentStorageService.write("/manifest", fakeSubmissionData.Manifest.getXmlData());
        documentStorageService.write("/submission", fakeSubmissionData.Xml.getXmlData());
        documentStorageService.write("/userContext", fakeSubmissionData.UserContext.toString());

        String submissionId = "123457abc";
        subject.addSubmission(
                previousSubmissionBatch,
                new UserSubmission("userId", "taxReturnId", submissionId, "/manifest", "/submission", "/userContext"));

        // Move clock forward to simulate time passing between each batch because the testClock uses fixed time.
        testClock.fastForward(Duration.of(batchingProperties.getBatchTimeoutMilliseconds() + 1000, ChronoUnit.MILLIS));
        subject.addSubmission(
                currentSubmissionBatch,
                new UserSubmission("userId", "taxReturnId", submissionId, "/manifest", "/submission", "/userContext"));

        // Verify we identify the old batch that needs to be processed
        List<SubmissionBatch> result = subject.getUnprocessedBatches(APPLICATION_ID);
        assertEquals(1, result.size());
        assertEquals(100L, result.get(0).batchId());
        assertEquals(
                DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, previousBatch, getBatchControlYear()),
                result.get(0).path());

        // Verify that the batch has one submission, and the path matches what we expect
        List<String> submissionsForOldBatch =
                documentStorageService.getSubFolders(DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, previousBatch, getBatchControlYear()));
        assertEquals(1, submissionsForOldBatch.size());

        String expectedPathForSubmission = DocumentStorageBatchRepository.generateLocationForBatch(
                        APPLICATION_ID, previousBatch, getBatchControlYear())
                + submissionId + "/";
        assertEquals(expectedPathForSubmission, submissionsForOldBatch.get(0));
    }

    @Test
    public void getUnprocessedBatches_givenNoSubFolders_returnsEmptyList() throws Exception {
        List<SubmissionBatch> result = subject.getUnprocessedBatches(APPLICATION_ID);
        assertEquals(0, result.size());
    }

    private int getBatchControlYear() {
        return Year.now(testClock).getValue() - 1;
    }
}
