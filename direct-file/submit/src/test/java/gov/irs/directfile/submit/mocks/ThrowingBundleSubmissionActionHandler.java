package gov.irs.directfile.submit.mocks;

import gov.irs.directfile.submit.actions.exception.SubmissionFailureException;
import gov.irs.directfile.submit.command.SubmitBundleAction;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.domain.BundledArchives;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.SubmittedDataContainer;
import gov.irs.directfile.submit.service.interfaces.IBundleSubmissionActionHandler;

public class ThrowingBundleSubmissionActionHandler implements IBundleSubmissionActionHandler {
    private boolean loginEnabled = true;
    private boolean logoutEnabled = true;

    @Override
    public boolean login() {
        return loginEnabled;
    }

    @Override
    public boolean logout() {
        return logoutEnabled;
    }

    @Override
    public SubmittedDataContainer submitBundles(BundledArchives bundledArchives, SubmissionBatch submissionBatch)
            throws SubmissionFailureException {
        throw new SubmissionFailureException(submissionBatch, bundledArchives, new RuntimeException("Can't submit"));
    }

    @Override
    public SubmittedDataContainer handleCommand(SubmitBundleAction command) throws SubmissionFailureException {
        return this.submitBundles(
                command.getBundleArchivesActionResult().getBundledArchives(),
                command.getBundleArchivesActionResult().getBatch());
    }

    @Override
    public void Setup(Config config) throws Throwable {}

    public void enableLogin() {
        this.loginEnabled = true;
    }

    public void disableLogin() {
        this.logoutEnabled = false;
    }

    public void enableLogout() {
        this.logoutEnabled = true;
    }

    public void disableLogout() {
        this.logoutEnabled = false;
    }
}
