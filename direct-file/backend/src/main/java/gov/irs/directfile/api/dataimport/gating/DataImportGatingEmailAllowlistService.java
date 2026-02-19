package gov.irs.directfile.api.dataimport.gating;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.pdfbox.util.Hex;
import org.bouncycastle.crypto.digests.SHA256Digest;
import org.bouncycastle.crypto.macs.HMac;
import org.bouncycastle.crypto.params.KeyParameter;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.config.DataImportGatingConfigurationProperties;

@Slf4j
@Service
public class DataImportGatingEmailAllowlistService {
    private final DataImportGatingConfigService dataImportGatingConfigService;

    @Getter
    private final boolean allowlistEnabled;

    private Set<String> allowlist;
    private final String allowListObject;
    private final byte[] hexKey;

    public DataImportGatingEmailAllowlistService(
            DataImportGatingConfigurationProperties configProps,
            DataImportGatingConfigService dataImportGatingConfigService) {
        this.dataImportGatingConfigService = dataImportGatingConfigService;
        this.allowlistEnabled = configProps.getAllowlist().enabled();
        this.allowListObject = configProps.getAllowlist().objectKey();
        this.hexKey = Hex.decodeHex(configProps.getAllowlist().hexKey());
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
        try {
            this.allowlist = Arrays.stream(dataImportGatingConfigService
                            .getDataImportGatingObjectAsString(allowListObject)
                            .split("\n"))
                    .collect(Collectors.toSet());
            log.info("Allowlist checked, total items: {}", allowlist.size());
        } catch (Exception e) {
            // should we set up an alert on this error?
            log.error("Error during allowlist retrieval: {}", e.getMessage());
            this.allowlist = Collections.emptySet();
        }
    }
}
