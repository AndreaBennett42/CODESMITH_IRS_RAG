package gov.irs.directfile.submit.service.interfaces;

import gov.irs.directfile.submit.actions.results.SubmissionFailureActionResult;
import gov.irs.directfile.submit.command.SubmissionFailureAction;
import gov.irs.directfile.submit.domain.SubmissionBatch;

public interface ISubmissionFailureService extends IService {
    void processFailedBatch(SubmissionBatch submissionBatch);

    SubmissionFailureActionResult handleCommand(SubmissionFailureAction submissionFailureActionCommand);
}
