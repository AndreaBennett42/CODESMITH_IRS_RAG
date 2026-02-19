package gov.irs.directfile.submit.actions.results;

import java.util.List;

import lombok.Getter;

import gov.irs.directfile.submit.domain.SubmissionArchiveContainer;
import gov.irs.directfile.submit.domain.SubmissionBatch;

@Getter
public class CreateArchiveActionResult {
    private SubmissionBatch batch;
    private List<SubmissionArchiveContainer> submissionArchiveContainers;

    public CreateArchiveActionResult(
            SubmissionBatch batch, List<SubmissionArchiveContainer> submissionArchiveContainers) {
        this.batch = batch;
        this.submissionArchiveContainers = submissionArchiveContainers;
    }
}
