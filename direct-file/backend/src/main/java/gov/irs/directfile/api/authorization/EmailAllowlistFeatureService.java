package gov.irs.directfile.api.authorization;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.bouncycastle.crypto.digests.SHA256Digest;
import org.bouncycastle.crypto.macs.HMac;
import org.bouncycastle.crypto.params.KeyParameter;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.authorization.config.FeatureFlagConfigurationProperties;
import gov.irs.directfile.api.featureflags.FeatureFlagService;

@Slf4j
@Service
public class EmailAllowlistFeatureService {
    private final FeatureFlagService featureFlagService;

    @Getter
    private final boolean allowlistEnabled;

    private Set<String> allowlist;
    private final String allowListObject;
    private final byte[] hexKey;

    public EmailAllowlistFeatureService(
            FeatureFlagConfigurationProperties configProps, FeatureFlagService featureFlagService) {
        this.featureFlagService = featureFlagService;
        this.allowlistEnabled = configProps.getAllowlist().enabled();
        this.allowListObject = configProps.getAllowlist().objectKey();
        this.hexKey = null;
    }

    // determines whether the identity provider-supplied email address is on our allowlist
    public boolean emailOnAllowlist(String email) {
        if (allowlistEnabled) {
            loadAllowlist(); // trigger cache reload if needed
            String base64Mac = emailMac(email);
            return allowlist.contains(base64Mac);
        }

        // allowlist disabled
        log.info("Allowlist is disabled, so emailOnAllowlist is false");
        return false;
    }

    private String emailMac(String email) {
        HMac hMac = new HMac(new SHA256Digest());
        hMac.init(new KeyParameter(hexKey));
        byte[] in = StringUtils.lowerCase(email).getBytes(StandardCharsets.UTF_8);
        hMac.update(in, 0, in.length);
        byte[] hMacOut = new byte[hMac.getMacSize()];
        hMac.doFinal(hMacOut, 0);
        return Base64.getEncoder().encodeToString(hMacOut);
    }

    private void loadAllowlist() {
        if (!allowlistEnabled) {
            return;
        }
        try {
            this.allowlist = Arrays.stream(featureFlagService
                            .getFeatureObjectAsString(allowListObject)
                            .split("\n"))
                    .collect(Collectors.toSet());
            log.info("Allowlist checked, total items: {}", allowlist.size());
        } catch (Exception e) {
            log.error("Error during allowlist retrieval: {}", e.getMessage());
            this.allowlist = Collections.emptySet();
        }
    }
}
