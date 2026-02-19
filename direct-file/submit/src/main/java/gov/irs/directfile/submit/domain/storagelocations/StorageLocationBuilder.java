package gov.irs.directfile.submit.domain.storagelocations;

import java.util.UUID;

public class StorageLocationBuilder {
    private static final String S3_SUBMISSIONS_FOLDER = "pre-submission-batching";

    public static String getTaxReturnLocation(int taxFilingYear, UUID taxReturnId) {
        return taxFilingYear + "/taxreturns/" + taxReturnId;
    }

    public static String getSubmissionLocation(
            int taxFilingYear, UUID taxReturnId, String submissionId, String fileExtension) {
        return getSubmissionLocation(taxFilingYear, taxReturnId) + submissionId + "." + fileExtension;
    }

    public static String getSubmissionLocation(int taxFilingYear, UUID taxReturnId) {
        return getTaxReturnLocation(taxFilingYear, taxReturnId) + "/submissions/";
    }

    public static String generateObjectKeyPrefixForSubmission(
            long batchId, String submissionId, String applicationId, int batchControlYear) {
        // {bucketName}/{configurable-application-id}/{batchControlYear}/{batch-number}/{submission-id}/DATA-GOES-HERE
        return String.format(
                "%s/%s/%s/%s/%s", S3_SUBMISSIONS_FOLDER, applicationId, batchControlYear, batchId, submissionId);
    }

    public static String getErrorFolderLocation(String applicationId, int batchControlYear) {
        return String.format("%s/errors/%s/%s/", S3_SUBMISSIONS_FOLDER, applicationId, batchControlYear);
    }
}
