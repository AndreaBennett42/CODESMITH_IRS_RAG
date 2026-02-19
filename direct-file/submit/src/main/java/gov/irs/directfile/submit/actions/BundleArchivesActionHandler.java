package gov.irs.directfile.submit.actions;

import java.util.List;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.mef.inputcomposition.PostmarkedSubmissionArchive;
import gov.irs.mef.inputcomposition.SubmissionBuilder;
import gov.irs.mef.inputcomposition.SubmissionContainer;

import gov.irs.directfile.audit.AuditService;
import gov.irs.directfile.submit.actions.exception.BundleArchiveActionException;
import gov.irs.directfile.submit.actions.results.BundleArchivesActionResult;
import gov.irs.directfile.submit.command.BundleArchiveAction;
import gov.irs.directfile.submit.domain.BundledArchives;
import gov.irs.directfile.submit.domain.SubmissionArchiveContainer;
import gov.irs.directfile.submit.domain.UserContextData;

@Slf4j
@SuppressFBWarnings(
        value = {"NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@Service
public class BundleArchivesActionHandler {

    private final ActionContext actionContext;

    public BundleArchivesActionHandler(ActionContext actionContext) {
        this.actionContext = actionContext;
    }

    private static final AuditService auditService = new AuditService();

    public BundleArchivesActionResult handleBundleCommand(BundleArchiveAction bundleArchiveActionCommand)
            throws BundleArchiveActionException {
        SubmissionContainer container = null;
        List<SubmissionArchiveContainer> archives =
                bundleArchiveActionCommand.getCreateArchiveActionResult().getSubmissionArchiveContainers();

        try {

            // TODO: what is the maximum bundle size?
            PostmarkedSubmissionArchive[] archs = new PostmarkedSubmissionArchive[archives.size()];
            for (int i = 0; i < archives.size(); i++) {
                archs[i] = archives.get(i).Archive;
            }
            container = SubmissionBuilder.createSubmissionContainer(
                    archs, actionContext.getConfig().getDirectories().getBatched());

        } catch (Exception e) {
            throw new BundleArchiveActionException(
                    getUserContextDataList(archives),
                    bundleArchiveActionCommand.getCreateArchiveActionResult().getBatch(),
                    e);
        }

        return new BundleArchivesActionResult(
                bundleArchiveActionCommand.getCreateArchiveActionResult().getBatch(),
                new BundledArchives(getUserContextDataList(archives), container));
    }

    private List<UserContextData> getUserContextDataList(List<SubmissionArchiveContainer> archives) {
        return archives.stream().map(x -> x.UserContext).toList();
    }
}
