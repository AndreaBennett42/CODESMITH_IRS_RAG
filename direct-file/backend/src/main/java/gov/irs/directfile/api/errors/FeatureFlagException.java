package gov.irs.directfile.api.errors;

public class FeatureFlagException extends RuntimeException {
    public FeatureFlagException(String message, Throwable cause) {
        super(message, cause);
    }
}
