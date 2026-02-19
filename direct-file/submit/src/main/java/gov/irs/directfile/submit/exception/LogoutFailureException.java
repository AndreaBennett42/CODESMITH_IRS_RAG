package gov.irs.directfile.submit.exception;

public class LogoutFailureException extends Exception {
    public LogoutFailureException(String message) {
        super(message);
    }

    public LogoutFailureException(String message, Throwable cause) {
        super(message, cause);
    }

    public LogoutFailureException(Throwable cause) {
        super(cause);
    }
}
