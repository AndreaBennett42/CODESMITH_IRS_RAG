package gov.irs.directfile.submit.command;

import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.directfile.submit.actions.exception.SubmissionFailureException;

@Getter
@AllArgsConstructor
public class SubmissionFailureAction extends Action {
    private final SubmissionFailureException submissionFailureException;

    @Override
    public ActionType getType() {
        return ActionType.SUBMISSION_FAILURE;
    }
}
