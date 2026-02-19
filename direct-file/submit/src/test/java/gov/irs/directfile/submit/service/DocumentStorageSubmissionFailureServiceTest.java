package gov.irs.directfile.submit.service;

import java.time.Clock;
import java.time.Year;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.mocks.FakeSynchronousDocumentStorageService;
import gov.irs.directfile.submit.mocks.MutableTestClock;

import static org.junit.jupiter.api.Assertions.assertNotNull;

@ExtendWith(MockitoExtension.class)
public class DocumentStorageSubmissionFailureServiceTest {
    private static final String APPLICATION_ID = "APPLICATION_ID";
    private static final Clock clock = new MutableTestClock();
    DocumentStorageSubmissionFailureService subject;
    FakeSynchronousDocumentStorageService fakeSynchronousDocumentStorageService =
            new FakeSynchronousDocumentStorageService();

    @BeforeEach
    public void setup() {
        subject = new DocumentStorageSubmissionFailureService(
                fakeSynchronousDocumentStorageService, APPLICATION_ID, clock);
    }

    @AfterEach
    public void teardown() {
        fakeSynchronousDocumentStorageService.clear();
    }

    @Test
    public void itWritesSubmissionFilesToCorrectDocumentStoragePath() throws Exception {
        int batchControlYear = Year.now(clock).getValue() - 1;
        long batchId = 10L;
        String submissionId = "subId-ABC";
        String submissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, batchId);

        addFakeSubmission(submissionBatchObjectKey, submissionId);
        SubmissionBatch batch = new SubmissionBatch(batchId, submissionBatchObjectKey);

        // 2. Call the process error batch
        subject.processFailedBatch(batch);

        // 3. We treat each submission as an individual batch. Validate all the files exist at a new error path for the
        // 0th item of the batch
        String errorPath = "pre-submission-batching/errors/APPLICATION_ID/" + batchControlYear + "/" + batch.batchId()
                + "/" + 0 + "/" + submissionId + "/";
        List<String> errorFolders = fakeSynchronousDocumentStorageService.getSubFolders(errorPath);

        Assertions.assertEquals(1, errorFolders.size());
        Assertions.assertEquals(errorPath, errorFolders.get(0));

        // Validate that we created files at the new error file paths. We know there is one item in this batch, so it
        // will be the 0th error batch of the original batch with batch id 10
        String submissionXmlErrorPath = String.format(
                "pre-submission-batching/errors/APPLICATION_ID/%s/10/0/subId-ABC/submission.xml", batchControlYear);
        String manifestXmlErrorPath = String.format(
                "pre-submission-batching/errors/APPLICATION_ID/%s/10/0/subId-ABC/manifest.xml", batchControlYear);
        String userContextJsonErrorPath = String.format(
                "pre-submission-batching/errors/APPLICATION_ID/%s/10/0/subId-ABC/userContext.json", batchControlYear);

        assertNotNull(fakeSynchronousDocumentStorageService.getObjectAsString(submissionXmlErrorPath));
        assertNotNull(fakeSynchronousDocumentStorageService.getObjectAsString(manifestXmlErrorPath));
        assertNotNull(fakeSynchronousDocumentStorageService.getObjectAsString(userContextJsonErrorPath));
    }

    @Test
    public void itCopiesEachSubmissionIntoAnErrorPathWithAUniqueBatchId() {
        // 1. Add a batch with some paths to the fake synchronous store
        int batchControlYear = Year.now(clock).getValue() - 1;
        long batchId = 10L;

        List<String> submissionIds = List.of("subId-ABC", "subId-DEF", "subId-GHI");
        String submissionBatchObjectKey =
                String.format("pre-submission-batching/%s/%s/%s/", APPLICATION_ID, batchControlYear, batchId);

        for (String submissionId : submissionIds) {
            addFakeSubmission(submissionBatchObjectKey, submissionId);
        }

        SubmissionBatch batch = new SubmissionBatch(batchId, submissionBatchObjectKey);
        // 2. Call the process error batch
        subject.processFailedBatch(batch);

        // 3. Because we'll treat each submission as an individual batch, we expect to find 3 subfolders for the path.
        String errorPath = String.format(
                "pre-submission-batching/errors/%s/%s/%s/", APPLICATION_ID, batchControlYear, batch.batchId());
        List<String> errorFolders = fakeSynchronousDocumentStorageService.getSubFolders(errorPath);

        Assertions.assertEquals(3, errorFolders.size());
    }

    private void addFakeSubmission(String batchObjectKey, String submissionId) {
        fakeSynchronousDocumentStorageService.write(
                batchObjectKey + submissionId + "/submission.xml", "submission.xml");
        fakeSynchronousDocumentStorageService.write(batchObjectKey + submissionId + "/manifest.xml", "manifest.xml");
        fakeSynchronousDocumentStorageService.write(
                batchObjectKey + submissionId + "/userContext.json", "userContext.json");
    }
}
