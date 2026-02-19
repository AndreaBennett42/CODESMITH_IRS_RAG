package gov.irs.directfile.submit;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.models.Dispatch;

import static org.junit.jupiter.api.Assertions.assertEquals;

// TODO: Delete this class after libs story (https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/1625) is
// merged.
public class LibsProofOfConceptTest {

    @Test
    public void useDispatchClassFromSharedLibs() {
        Dispatch dispatch = new Dispatch();
        dispatch.setMefSubmissionId("111");

        assertEquals("111", dispatch.getMefSubmissionId());
    }
}
