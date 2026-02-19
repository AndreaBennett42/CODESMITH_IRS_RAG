package gov.irs.directfile.api.errors;

public class InvalidOperationException extends Exception {

    public InvalidOperationException() {}

    public InvalidOperationException(String message) {
        super(message);
    }
}
