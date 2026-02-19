package gov.irs.directfile.submit.exception;

public class LoginFailureException extends Exception {
    public LoginFailureException(String message) {
        super(message);
    }

    public LoginFailureException(String message, Throwable cause) {
        super(message, cause);
    }

    public LoginFailureException(Throwable cause) {
        super(cause);
    }
}
