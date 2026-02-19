package gov.irs.directfile.models.message.pending;

import java.util.HashMap;
import java.util.Map;

import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum PendingSubmissionMessageVersion {
    V1("1.0");

    private final String version;
    private static final Map<String, PendingSubmissionMessageVersion> versionMap = new HashMap<>();

    static {
        for (PendingSubmissionMessageVersion v : PendingSubmissionMessageVersion.values()) {
            versionMap.put(v.getVersion(), v);
        }
    }

    PendingSubmissionMessageVersion(String version) {
        this.version = version;
    }

    public String getVersion() {
        return version;
    }

    public static PendingSubmissionMessageVersion getEnum(String version) {
        PendingSubmissionMessageVersion result = versionMap.get(version);
        if (result == null) {
            throw new UnsupportedVersionException("No enum found for version: " + version);
        }
        return result;
    }
}
