package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class KeystoreConfig {
    private String keystoreBase64;
    private String keystorePassword;
    private String keystoreAlias;
}
