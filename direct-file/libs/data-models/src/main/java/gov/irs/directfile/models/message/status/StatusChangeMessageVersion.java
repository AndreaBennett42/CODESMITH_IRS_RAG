package gov.irs.directfile.models.message.status;

import java.util.HashMap;
import java.util.Map;

import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum StatusChangeMessageVersion {
    V1("1.0");

    private final String version;
    private static final Map<String, StatusChangeMessageVersion> versionMap = new HashMap<>();

    static {
        for (StatusChangeMessageVersion v : StatusChangeMessageVersion.values()) {
            versionMap.put(v.getVersion(), v);
        }
    }

    StatusChangeMessageVersion(String version) {
        this.version = version;
    }

    public String getVersion() {
        return version;
    }

    public static StatusChangeMessageVersion getEnum(String version) {
        StatusChangeMessageVersion result = versionMap.get(version);
        if (result == null) {
            throw new UnsupportedVersionException("No enum found for version: " + version);
        }
        return result;
    }
}
