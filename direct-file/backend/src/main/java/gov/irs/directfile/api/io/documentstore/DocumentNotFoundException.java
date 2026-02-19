package gov.irs.directfile.api.io.documentstore;

public class DocumentNotFoundException extends DocumentStoreException {
    public DocumentNotFoundException(String message) {
        super(message);
    }

    public DocumentNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}
