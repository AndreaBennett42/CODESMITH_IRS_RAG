package gov.irs.directfile.submit.domain;

import java.util.UUID;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.Dispatch;

import static org.junit.jupiter.api.Assertions.assertEquals;

class UserSubmissionTest {
    @Test
    public void fromDispatch_success() {
        Dispatch dispatch = new Dispatch(
                UUID.randomUUID(),
                UUID.randomUUID(),
                "/path/to/manifest",
                "/path/to/userContext",
                "/path/to/submission",
                "123456789");

        UserSubmission userSubmission = UserSubmission.fromDispatch(dispatch);

        assertEquals(dispatch.getUserId().toString(), userSubmission.userId());
        assertEquals(dispatch.getTaxReturnId().toString(), userSubmission.taxReturnId());
        assertEquals(dispatch.getMefSubmissionId(), userSubmission.submissionId());
        assertEquals(dispatch.getPathToManifest(), userSubmission.manifestXmlPath());
        assertEquals(dispatch.getPathToSubmission(), userSubmission.submissionXmlPath());
        assertEquals(dispatch.getPathToUserContext(), userSubmission.userContextPath());
    }
}
