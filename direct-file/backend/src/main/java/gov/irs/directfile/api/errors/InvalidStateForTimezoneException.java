package gov.irs.directfile.api.errors;

public class InvalidStateForTimezoneException extends Exception {
    public InvalidStateForTimezoneException(String message) {
        super(message);
    }
}
