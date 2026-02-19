package gov.irs.directfile.submit.actions.exception;

import lombok.Getter;

import gov.irs.directfile.submit.actions.ActionException;
import gov.irs.directfile.submit.domain.BundledArchives;
import gov.irs.directfile.submit.domain.SubmissionBatch;

@Getter
public class SubmissionFailureException extends ActionException {
    private SubmissionBatch batch;
    private BundledArchives bundledArchives;

    public SubmissionFailureException(SubmissionBatch batch, BundledArchives archives, Throwable e) {
        super(e);
        this.batch = batch;
        this.bundledArchives = archives;
    }
}
