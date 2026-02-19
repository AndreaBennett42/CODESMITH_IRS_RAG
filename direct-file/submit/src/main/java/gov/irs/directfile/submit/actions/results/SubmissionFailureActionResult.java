package gov.irs.directfile.submit.actions.results;

import lombok.Getter;

import gov.irs.directfile.submit.domain.SubmissionBatch;

@Getter
public class SubmissionFailureActionResult {
    private SubmissionBatch batch;

    public SubmissionFailureActionResult(SubmissionBatch batch) {
        this.batch = batch;
    }
}
