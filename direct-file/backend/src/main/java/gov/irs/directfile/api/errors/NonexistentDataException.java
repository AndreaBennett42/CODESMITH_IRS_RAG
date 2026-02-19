package gov.irs.directfile.api.errors;

public class NonexistentDataException extends RuntimeException {

    public NonexistentDataException() {
        super();
    }

    public NonexistentDataException(String message) {
        super(message);
    }
}
