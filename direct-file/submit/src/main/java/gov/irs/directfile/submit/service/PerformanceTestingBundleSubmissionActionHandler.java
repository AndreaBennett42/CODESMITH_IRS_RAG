package gov.irs.directfile.submit.service;

import java.time.Duration;
import java.util.*;
import javax.xml.datatype.XMLGregorianCalendar;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.actions.ActionContext;
import gov.irs.directfile.submit.actions.exception.SubmissionFailureException;
import gov.irs.directfile.submit.command.SubmitBundleAction;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.domain.*;
import gov.irs.directfile.submit.exception.LoginFailureException;
import gov.irs.directfile.submit.exception.LogoutFailureException;
import gov.irs.directfile.submit.service.interfaces.IBundleSubmissionActionHandler;

@Service
/**
 * A Bundle Submission Service used for perf testing.
 *
 * This generates fake submission receipts so we can easily
 * verify if the submit app is processing all the returns it receives
 * by checking that the backend app receives an equivalent number of submission confirmations.
 * */
@SuppressFBWarnings(
        value = {"NM_METHOD_NAMING_CONVENTION", "PREDICTABLE_RANDOM"},
        justification =
                "Initial SpotBugs Setup. Random usage is not for security purposes and only used in performance testing.")
@Slf4j
public class PerformanceTestingBundleSubmissionActionHandler implements IBundleSubmissionActionHandler {
    private final Config config;
    private final ActionContext context;
    private final Random random = new Random();
    private static final long TEN_SECONDS_IN_MILLIS = 10_000;

    public PerformanceTestingBundleSubmissionActionHandler(Config config, ActionContext context) {
        log.info("Initializing Mock Bundle Submission Service for Performance Testing.");
        this.config = config;
        this.context = context;
    }

    @Override
    public boolean login() throws LoginFailureException {
        return true;
    }

    @Override
    public boolean logout() throws LogoutFailureException {
        return true;
    }

    @Override
    public SubmittedDataContainer submitBundles(BundledArchives bundledArchives, SubmissionBatch submissionBatch)
            throws SubmissionFailureException {
        SendSubmissionsResultWrapper fakeSubmissionResultWrapper = generateMockData(bundledArchives);
        try {

            long jitter = random.nextLong(2000);
            Thread.sleep(Duration.ofMillis(TEN_SECONDS_IN_MILLIS + jitter));
            log.info(
                    "Using Mock Bundle Submission Service for Performance Testing. Creating Mock MEF Response for batch {}",
                    submissionBatch);
            return new SubmittedDataContainer(
                    bundledArchives.UserContexts, fakeSubmissionResultWrapper, submissionBatch);
        } catch (Exception e) {
            log.error("Exception in Mock Bundle Submission Action Handler");
            return new SubmittedDataContainer(
                    bundledArchives.UserContexts, fakeSubmissionResultWrapper, submissionBatch);
        }
    }

    private static SendSubmissionsResultWrapper generateMockData(BundledArchives bundledArchives) {
        List<SubmissionReceiptGrpWrapper> fakeSubmissionReceiptWrappers = new ArrayList<>();
        bundledArchives.UserContexts.forEach(userContextData -> {
            String fakeReceiptId = UUID.randomUUID().toString();
            XMLGregorianCalendar calendar =
                    LocalWriteUtilityService.CreateGregorianDateFromString(userContextData.getSignDate());
            fakeSubmissionReceiptWrappers.add(
                    new SubmissionReceiptGrpWrapper(userContextData.getSubmissionId(), fakeReceiptId, calendar));
        });

        SendSubmissionsResultWrapper fakeSubmissionResultWrapper =
                new SendSubmissionsResultWrapper(fakeSubmissionReceiptWrappers);
        return fakeSubmissionResultWrapper;
    }

    @Override
    public SubmittedDataContainer handleCommand(SubmitBundleAction command) throws SubmissionFailureException {
        return this.submitBundles(
                command.getBundleArchivesActionResult().getBundledArchives(),
                command.getBundleArchivesActionResult().getBatch());
    }

    @Override
    public void Setup(Config config) throws Throwable {}
}
