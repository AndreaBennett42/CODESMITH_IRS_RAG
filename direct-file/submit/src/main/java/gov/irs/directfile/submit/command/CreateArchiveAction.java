package gov.irs.directfile.submit.command;

import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.directfile.submit.domain.SubmissionBatch;

@Getter
@AllArgsConstructor
public class CreateArchiveAction extends Action {
    private final SubmissionBatch submissionBatch;

    @Override
    public ActionType getType() {
        return ActionType.CREATE_ARCHIVE;
    }
}
