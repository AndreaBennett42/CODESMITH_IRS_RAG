package gov.irs.directfile.submit.service.interfaces;

import gov.irs.directfile.submit.actions.exception.SubmissionFailureException;
import gov.irs.directfile.submit.command.SubmitBundleAction;
import gov.irs.directfile.submit.domain.BundledArchives;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.SubmittedDataContainer;
import gov.irs.directfile.submit.exception.LoginFailureException;
import gov.irs.directfile.submit.exception.LogoutFailureException;

public interface IBundleSubmissionActionHandler extends IService {

    boolean login() throws LoginFailureException;

    boolean logout() throws LogoutFailureException;

    SubmittedDataContainer submitBundles(BundledArchives bundledArchives, SubmissionBatch submissionBatch)
            throws SubmissionFailureException;

    SubmittedDataContainer handleCommand(SubmitBundleAction command) throws SubmissionFailureException;
}
