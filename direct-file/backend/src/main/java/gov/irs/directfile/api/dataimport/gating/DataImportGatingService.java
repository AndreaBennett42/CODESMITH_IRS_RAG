package gov.irs.directfile.api.dataimport.gating;

import java.security.SecureRandom;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.config.DataImportGatingConfigurationProperties;
import gov.irs.directfile.api.config.DataImportGatingS3Properties;

@Slf4j
@Service
@EnableConfigurationProperties({DataImportGatingS3Properties.class, DataImportGatingConfigurationProperties.class})
public class DataImportGatingService {

    @Value("${direct-file.data-import.enabled}")
    private boolean dataImportEnabled;

    private final DataImportGatingConfigService gatingConfigService;
    private final DataImportGatingEmailAllowlistService emailAllowlistService;

    private static final SecureRandom secureRandom = new SecureRandom();

    public DataImportGatingService(
            DataImportGatingConfigService gatingConfigService,
            DataImportGatingEmailAllowlistService emailAllowlistService) {
        this.gatingConfigService = gatingConfigService;
        this.emailAllowlistService = emailAllowlistService;
    }

    public DataImportBehavior getBehavior(String currentUserEmail) {
        if (!dataImportEnabled) {
            return DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC;
        }

        // if allow-list disabled or not on allow-list, will continue with windowing
        if (emailAllowlistService.emailOnAllowlist(currentUserEmail)) {
            return DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2;
        }

        DataImportGatingConfig config = gatingConfigService.getGatingS3Config();
        if (config == null) {
            log.warn(
                    "Failed to retrieve data-import-gating config file, default to DataImportBehavior DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN");
            return DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN;
        }

        // if windowing is empty, assume all time available
        if (config.getWindowing().isEmpty()) {
            return pickPercentageBasedBehavior(config);
        } else {
            ZonedDateTime now = getCurrentTime();
            boolean inWindow = config.getWindowing().stream()
                    .anyMatch(window -> now.isAfter(window.getStart()) && now.isBefore(window.getEnd()));

            if (inWindow) {
                return pickPercentageBasedBehavior(config);
            } else {
                log.warn(
                        "Data import not allowed outside the specified time windows: {}",
                        config.getWindowing().toString());
                return DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN;
            }
        }
    }

    private DataImportBehavior pickPercentageBasedBehavior(DataImportGatingConfig config) {
        // if percentage empty, assume default behavior (3)
        if (config.getPercentages().isEmpty()) {
            return DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN;
        }

        List<String> behaviorPool = new ArrayList<>();
        int totalPercentage = 0;

        // Populate the behavior pool based on percentages
        for (DataImportGatingConfig.Percentage percentage : config.getPercentages()) {
            int count = percentage.getPercentage();
            if (count > 0) {
                behaviorPool.addAll(Collections.nCopies(count, percentage.getBehavior()));
            }
            totalPercentage += count;
        }

        // Fill remaining percentage with default behavior
        DataImportBehavior defaultBehavior = DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN;
        if (totalPercentage < 100) {
            behaviorPool.addAll(Collections.nCopies(100 - totalPercentage, defaultBehavior.name()));
        }

        // Shuffle and pick a random behavior
        Collections.shuffle(behaviorPool);
        String behaviorStr = behaviorPool.get(secureRandom.nextInt(behaviorPool.size()));

        log.info("pickPercentageBasedBehavior: {}", behaviorStr);
        return DataImportBehavior.valueOf(behaviorStr);
    }

    protected ZonedDateTime getCurrentTime() {
        return ZonedDateTime.now();
    }
}
