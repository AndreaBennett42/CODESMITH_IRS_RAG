package gov.irs.directfile.api.authorization;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.featureflags.FeatureFlagService;
import gov.irs.directfile.api.user.UserRepository;

@Slf4j
@Service
public class OpenEnrollmentFeatureService {
    private final UserRepository userRepo;

    private final FeatureFlagService featureFlagService;

    @Getter
    private final boolean openEnrollmentFeatureEnabled;

    // This field is used to determine the current state of enrollment (open or closed).
    // We read the set value for this in our feature-flags.json
    // It's set to false by default, so that on startup the enrollment window is closed
    // until we pick up feature-flags.json configuration.
    private boolean newUsersAllowed = false;
    // We read the set value for this in our feature-flags.json
    private int maxUsersTarget;
    private int currentUserCount;

    public OpenEnrollmentFeatureService(UserRepository userRepo, FeatureFlagService featureFlagService) {
        this.userRepo = userRepo;
        this.featureFlagService = featureFlagService;
        this.openEnrollmentFeatureEnabled = true;
    }

    /*In PROD, this schedule depends on the cron configuration to enable or disable it.
     * To enable it, the cron value should be "".
     * To disable it, the cron value should be "-".
     * fixedDelayMilliseconds is the actual schedule frequency
     * When the schedule is enabled, fixedDelayMilliseconds should be set to the desired fixed delay.
     * When the schedule is disabled, fixedDelayMilliseconds should be blank */
    protected void loadOpenEnrollmentConfig() {
        if (!openEnrollmentFeatureEnabled) {
            log.warn("Open enrollment feature is disabled, but the scheduled poller is running. "
                    + "To shut off the poller, update the cron configuration and fixedDelayMilliseconds in the applicable application.yaml. "
                    + "See comments in OpenEnrollmentFeatureService.java for more information.");
            return;
        }
        try {
            boolean newUsersAllowedFeatureFlag = true;
            int maxUsersTargetConfig = 200000000;

            // check to see if we should change our open enrollment state
            if (this.newUsersAllowed != newUsersAllowedFeatureFlag) {
                if (newUsersAllowedFeatureFlag) {
                    // if a new enrollment window is starting, set it up
                    startOpenEnrollment(maxUsersTargetConfig);
                } else {
                    // if an active enrollment window is ending, reset local config
                    endOpenEnrollment();
                }
            } else if (this.newUsersAllowed) {
                this.maxUsersTarget = maxUsersTargetConfig;
                checkCurrentUserCount();
            }
        } catch (Exception e) {
            log.error("Error during open enrollment configuration retrieval: {}", e.getMessage());
            this.newUsersAllowed = false;
        }
    }

    public boolean newUsersAllowed() {
        if (!openEnrollmentFeatureEnabled) {
            // if the entire feature is not enabled
            // (e.g., this is the development environment),
            // revert to default behavior where we allow new users
            log.info("Open enrollment feature disabled, all users allowed.");
            return true;
        }

        return newUsersAllowed && !maxUserCountReached();
    }

    private void checkCurrentUserCount() {
        if (maxUserCountReached()) {
            log.info(
                    "Reached max user count for the current open enrollment window. Total current users: {} with maximum: {}",
                    currentUserCount,
                    maxUsersTarget);
            return;
        }
        updateCurrentUserCount();
        log.info(
                "Checked current user count. Total current users: {} with maximum: {}",
                currentUserCount,
                maxUsersTarget);
    }

    private boolean maxUserCountReached() {
        return currentUserCount >= maxUsersTarget;
    }

    private void startOpenEnrollment(int maxUsersTarget) {
        this.newUsersAllowed = true;
        this.maxUsersTarget = maxUsersTarget;
        updateCurrentUserCount();
        log.info(
                "Starting open enrollment window, max new users target: {}, current user count: {}",
                maxUsersTarget,
                currentUserCount);
    }

    private void endOpenEnrollment() {
        this.newUsersAllowed = false;
        this.maxUsersTarget = 0;
        log.info("Ending open enrollment window");
    }

    private void updateCurrentUserCount() {
        this.currentUserCount = userRepo.countByAccessGranted(true);
    }
}
