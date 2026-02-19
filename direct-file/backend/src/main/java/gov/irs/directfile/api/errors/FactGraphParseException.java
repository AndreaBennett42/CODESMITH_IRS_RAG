package gov.irs.directfile.api.errors;

public class FactGraphParseException extends RuntimeException {

    public FactGraphParseException() {}

    public FactGraphParseException(Throwable cause) {
        super(cause);
    }

    public FactGraphParseException(String message) {
        super(message);
    }

    public FactGraphParseException(String message, Throwable cause) {
        super(message, cause);
    }
}
