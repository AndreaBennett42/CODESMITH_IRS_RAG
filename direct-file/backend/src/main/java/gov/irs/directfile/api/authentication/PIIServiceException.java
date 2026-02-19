package gov.irs.directfile.api.authentication;

public class PIIServiceException extends RuntimeException {
    public PIIServiceException(Throwable cause) {
        super(cause);
    }

    public PIIServiceException(String message) {
        super(message);
    }

    public PIIServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
