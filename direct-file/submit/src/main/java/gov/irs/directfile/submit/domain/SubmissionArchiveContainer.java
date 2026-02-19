package gov.irs.directfile.submit.domain;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;

import gov.irs.mef.inputcomposition.PostmarkedSubmissionArchive;

@SuppressFBWarnings(
        value = {"NM_FIELD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
public class SubmissionArchiveContainer {
    public SubmissionArchiveContainer(UserContextData userContext, PostmarkedSubmissionArchive archive) {
        this.UserContext = userContext;
        this.Archive = archive;
    }

    public UserContextData UserContext;
    public PostmarkedSubmissionArchive Archive;
}
