package gov.irs.directfile.models.message.dispatch;

import java.util.HashMap;
import java.util.Map;

import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum DispatchMessageVersion {
    V1("1.0");

    private final String version;
    private static final Map<String, DispatchMessageVersion> versionMap = new HashMap<>();

    static {
        for (DispatchMessageVersion v : DispatchMessageVersion.values()) {
            versionMap.put(v.getVersion(), v);
        }
    }

    DispatchMessageVersion(String version) {
        this.version = version;
    }

    public String getVersion() {
        return version;
    }

    public static DispatchMessageVersion getEnum(String version) {
        DispatchMessageVersion result = versionMap.get(version);
        if (result == null) {
            throw new UnsupportedVersionException("No enum found for version: " + version);
        }
        return result;
    }
}
