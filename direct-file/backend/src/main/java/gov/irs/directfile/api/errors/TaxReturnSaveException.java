package gov.irs.directfile.api.errors;

public class TaxReturnSaveException extends RuntimeException {
    public TaxReturnSaveException(String message, Throwable cause) {
        super(message, cause);
    }
}
