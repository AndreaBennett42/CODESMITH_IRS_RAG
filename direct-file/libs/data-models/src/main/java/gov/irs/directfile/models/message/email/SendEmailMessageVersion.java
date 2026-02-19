package gov.irs.directfile.models.message.email;

import java.util.HashMap;
import java.util.Map;

import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum SendEmailMessageVersion {
    V1("1.0");

    private final String version;
    private static final Map<String, SendEmailMessageVersion> versionMap = new HashMap<>();

    static {
        for (SendEmailMessageVersion v : SendEmailMessageVersion.values()) {
            versionMap.put(v.getVersion(), v);
        }
    }

    SendEmailMessageVersion(String version) {
        this.version = version;
    }

    public String getVersion() {
        return version;
    }

    public static SendEmailMessageVersion getEnum(String version) {
        SendEmailMessageVersion result = versionMap.get(version);
        if (result == null) {
            throw new UnsupportedVersionException("No enum found for version: " + version);
        }
        return result;
    }
}
