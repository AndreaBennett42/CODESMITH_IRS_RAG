package gov.irs.directfile.submit.repository;

import java.io.ByteArrayInputStream;
import java.time.Clock;
import java.time.Instant;
import java.time.Year;
import java.util.List;
import java.util.Optional;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Repository;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.submit.BatchingProperties;
import gov.irs.directfile.submit.domain.DocumentStoreResource;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.UserSubmission;
import gov.irs.directfile.submit.domain.storagelocations.StorageLocationBuilder;
import gov.irs.directfile.submit.repository.interfaces.IBatchRepository;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@Slf4j
@Repository
@SuppressFBWarnings(
        value = {"DM_DEFAULT_ENCODING"},
        justification = "Initial SpotBugs Setup")
public class DocumentStorageBatchRepository implements IBatchRepository {
    private final ObjectMapper objectMapper = new ObjectMapper();
    private static final String S3_SUBMISSIONS_FOLDER = "pre-submission-batching";
    private final String applicationId;
    private final ISynchronousDocumentStoreService synchronousDocumentStoreService;
    private final BatchingProperties batchingProperties;
    private final Clock clock;

    @Autowired
    public DocumentStorageBatchRepository(
            ISynchronousDocumentStoreService synchronousDocumentStoreService,
            @Value("${submit.application-id}") String applicationId,
            BatchingProperties batchingProperties,
            Clock clock) {
        this.synchronousDocumentStoreService = synchronousDocumentStoreService;
        this.applicationId = applicationId;
        this.batchingProperties = batchingProperties;
        this.clock = clock;
    }

    @Override
    public long getCurrentWritingBatch(String applicationId) {
        // 1. S3 ls for  /{configurable-application-id}/{batchControlYear}/, store in variable
        int taxYear = getBatchControlYear();
        String taxYearS3Key = S3_SUBMISSIONS_FOLDER + "/" + applicationId + "/" + taxYear + "/";
        Optional<String> mostRecentBatchPrefixOptional =
                synchronousDocumentStoreService.getMostRecentFolderForPrefix(taxYearS3Key);

        // Case where no batches are found in s3
        if (mostRecentBatchPrefixOptional.isEmpty()) {
            return 0L;
        } else {
            // Case where a batch is found in s3
            String mostRecentBatchPrefix = mostRecentBatchPrefixOptional.get();

            long currentBatchNumber = extractBatchNumberFromPrefix(mostRecentBatchPrefix);
            List<String> userSubmissions = synchronousDocumentStoreService.getSubFolders(mostRecentBatchPrefix);
            boolean batchHasRecords = !userSubmissions.isEmpty();
            if (batchHasRecords) {
                // Get the timestamp of the least recently modified item in the batch. The age of a batch  = now() -
                // oldest upload time.
                String oldestSubmissionForBatch = userSubmissions.get(0);
                Instant lastModifiedTimestamp = synchronousDocumentStoreService
                        .getLeastRecentModifiedResourceForPrefix(oldestSubmissionForBatch)
                        .get()
                        .getLastModified();

                long batchAgeInMilliseconds = getBatchAge(lastModifiedTimestamp);
                boolean isBatchOlderThanTimeout =
                        batchAgeInMilliseconds >= batchingProperties.getBatchTimeoutMilliseconds();
                //  If the system finds a batch number, and the batch contains records, and the batch is older than the
                // timeout, or the # of records is > maxBatchSize
                // start a new batch, otherwise ret
                if (isBatchOlderThanTimeout || userSubmissions.size() >= batchingProperties.getMaxBatchSize()) {
                    // The system should create a new batch with a number one higher
                    return currentBatchNumber + 1;
                }
            }
            return currentBatchNumber;
        }
    }

    @Override
    public void addSubmission(SubmissionBatch submissionBatch, UserSubmission userSubmission) throws Exception {
        // Use the CreateXML action to generate XML for the userSubmission.

        // Get the documents from S3 artifact storage
        var manifest = synchronousDocumentStoreService.getObjectAsString(userSubmission.manifestXmlPath());
        var submission = synchronousDocumentStoreService.getObjectAsString(userSubmission.submissionXmlPath());
        var userContext = synchronousDocumentStoreService.getObjectAsString(userSubmission.userContextPath());

        // Upload to s3 batching area
        int taxYear = getBatchControlYear();
        String s3ObjectKey = StorageLocationBuilder.generateObjectKeyPrefixForSubmission(
                submissionBatch.batchId(), userSubmission.submissionId(), applicationId, taxYear);

        log.info(
                String.format("Uploading User Submission XML for user %s to %s", userSubmission.userId(), s3ObjectKey));
        // Upload XML, user context, and binary objects to s3. Write UserContext, and BinaryObjects as Bytes
        synchronousDocumentStoreService.write(
                s3ObjectKey + "/" + "manifest.xml", new ByteArrayInputStream(manifest.getBytes()));
        synchronousDocumentStoreService.write(
                s3ObjectKey + "/" + "submission.xml", new ByteArrayInputStream(submission.getBytes()));

        synchronousDocumentStoreService.write(
                s3ObjectKey + "/" + "userContext.json", new ByteArrayInputStream(userContext.getBytes()));
        MDC.put(AuditLogElement.taxReturnId.toString(), userSubmission.taxReturnId());
        MDC.put(AuditLogElement.mefSubmissionId.toString(), userSubmission.submissionId());
        log.info(String.format("Successfully wrote XML for user %s to %s", userSubmission.userId(), s3ObjectKey));
        MDC.clear();
    }

    @Override
    public Optional<SubmissionBatch> getSubmissionBatch(String applicationId, long batchId) {
        String path = generateLocationForBatch(applicationId, batchId, getBatchControlYear());
        List<DocumentStoreResource> keys = synchronousDocumentStoreService.getObjectKeys(path);
        if (keys.isEmpty()) {
            return Optional.empty();
        } else {
            return Optional.of(new SubmissionBatch(batchId, path));
        }
    }

    @Override
    public List<SubmissionBatch> getUnprocessedBatches(String applicationId) {
        long currentBatchNumber = this.getCurrentWritingBatch(applicationId);
        String path = generateLocationForApplicationId(applicationId, getBatchControlYear());

        List<String> subFolders = synchronousDocumentStoreService.getSubFolders(path);
        List<SubmissionBatch> submissionBatches = subFolders.stream()
                .map(subFolder -> extractBatchNumberFromPrefix(subFolder))
                .filter(batchNumber -> batchNumber != currentBatchNumber)
                .map(batchNumber -> new SubmissionBatch(
                        batchNumber, generateLocationForBatch(applicationId, batchNumber, getBatchControlYear())))
                .toList();
        return submissionBatches;
    }

    private static long extractBatchNumberFromPrefix(String mostRecentBatchPrefix) {
        String[] split = mostRecentBatchPrefix.split("/");
        return Long.parseLong(split[split.length - 1]);
    }

    private long getBatchAge(Instant lastModifiedTimestamp) {
        Instant currentTime = clock.instant();
        return currentTime.toEpochMilli() - lastModifiedTimestamp.toEpochMilli();
    }

    public static String generateLocationForBatch(String applicationId, long batchId, int batchControlYear) {
        return S3_SUBMISSIONS_FOLDER + "/" + applicationId + "/" + batchControlYear + "/" + batchId + "/";
    }

    public static String generateLocationForApplicationId(String applicationId, int batchControlYear) {
        return S3_SUBMISSIONS_FOLDER + "/" + applicationId + "/" + batchControlYear + "/";
    }

    /**
     * For purposes of uploading to s3, we use a specific path for the objects going to s3:
     *  {S3_SUBMISSIONS_FOLDER}/{configurable-application-id}/{batchControlYear}/{batchId}
     *
     * Today this is hardcoded to currentYear - 1, but in the future we will want
     * to support multiple different years for batches - a Year over Year Solution.
     *
     * Such a solution would likely require a few changes:
     * 1. Knowing when a UserSubmission was created. Possibly derive the year
     * from the dispatch object in the UserSubmissionMapper
     *
     * 2. Use the derived year to find the currentWriting batch for a given year.
     *
     * */
    private int getBatchControlYear() {
        return Year.now(clock).getValue() - 1;
    }
}
