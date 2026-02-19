package gov.irs.directfile.api.dataimport.exception;

public class DataImportException extends RuntimeException {
    public DataImportException() {}

    public DataImportException(String message) {
        super(message);
    }

    public DataImportException(String message, Throwable t) {
        super(message, t);
    }
}
