package gov.irs.directfile.submit.domain;

import gov.irs.directfile.models.Dispatch;

public record UserSubmission(
        String userId,
        String taxReturnId,
        String submissionId,
        String manifestXmlPath,
        String submissionXmlPath,
        String userContextPath) {
    public static UserSubmission fromDispatch(Dispatch dispatch) {
        return new UserSubmission(
                dispatch.getUserId().toString(),
                dispatch.getTaxReturnId().toString(),
                dispatch.getMefSubmissionId(),
                dispatch.getPathToManifest(),
                dispatch.getPathToSubmission(),
                dispatch.getPathToUserContext());
    }
}
