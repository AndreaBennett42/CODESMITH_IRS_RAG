package gov.irs.directfile.submit.domain;

import java.util.List;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;

import gov.irs.mef.inputcomposition.SubmissionContainer;

@SuppressFBWarnings(
        value = {"NM_FIELD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
public class BundledArchives {
    public BundledArchives(List<UserContextData> userContexts, SubmissionContainer submissionContainer) {
        UserContexts = userContexts;
        this.submissionContainer = submissionContainer;
    }

    public List<UserContextData> UserContexts;
    public SubmissionContainer submissionContainer;

    @Override
    public String toString() {
        return "BundledArchives{" + "SubmissionIds="
                + UserContexts.stream().map(UserContextData::getSubmissionId).toList() + '}';
    }
}
