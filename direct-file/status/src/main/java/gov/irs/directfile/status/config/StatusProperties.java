package gov.irs.directfile.status.config;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@SuppressFBWarnings(
        value = {"NM_FIELD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@AllArgsConstructor
@Getter
@Validated
@ConfigurationProperties(prefix = "status")
public class StatusProperties {
    @NotBlank
    private String applicationId;

    private String keystoreBase64;
    private String keystorePassword;
    private String keystoreAlias;
    private String etin;

    @Setter
    private String asid;

    private String efin;
    private boolean unitTesting;
    private boolean prod;
    private String rootTranslationKey;
    private String translationKeySplitter;
    public Long AckPollInMilliseconds;
    public boolean statusEndpointReturnsPendingByDefaultEnabled;
    private String toolkit;
    private boolean statusPollingEnabled;
}
