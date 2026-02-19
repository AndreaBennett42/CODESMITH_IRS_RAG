package gov.irs.directfile.models.message.confirmation;

import java.util.HashMap;
import java.util.Map;

import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum SubmissionConfirmationMessageVersion {
    /**
     * @deprecated V1 messages are no longer sent, but keeping version enum until we are sure queues are V1-free
     */
    @Deprecated
    V1("1.0"),
    V2("2.0");

    private final String version;
    private static final Map<String, SubmissionConfirmationMessageVersion> versionMap = new HashMap<>();

    static {
        for (SubmissionConfirmationMessageVersion v : SubmissionConfirmationMessageVersion.values()) {
            versionMap.put(v.getVersion(), v);
        }
    }

    SubmissionConfirmationMessageVersion(String version) {
        this.version = version;
    }

    public String getVersion() {
        return version;
    }

    public static SubmissionConfirmationMessageVersion getEnum(String version) {
        SubmissionConfirmationMessageVersion result = versionMap.get(version);
        if (result == null) {
            throw new UnsupportedVersionException("No enum found for version: " + version);
        }
        return result;
    }
}
