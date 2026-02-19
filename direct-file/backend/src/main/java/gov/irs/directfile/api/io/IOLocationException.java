package gov.irs.directfile.api.io;

public class IOLocationException extends Exception {
    public IOLocationException() {}

    public IOLocationException(String message) {
        super(message);
    }

    public IOLocationException(String message, Throwable cause) {
        super(message, cause);
    }
}
