package gov.irs.directfile.submit.repository.interfaces;

import java.util.List;
import java.util.Optional;

import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.UserSubmission;

@SuppressWarnings({"PMD.SignatureDeclareThrowsException", "PMD.UnnecessaryModifier"})
public interface IBatchRepository {
    public long getCurrentWritingBatch(String applicationId);

    public void addSubmission(SubmissionBatch submissionBatch, UserSubmission userSubmission) throws Exception;

    public Optional<SubmissionBatch> getSubmissionBatch(String applicationId, long batchId);

    public List<SubmissionBatch> getUnprocessedBatches(String applicationId);
}
