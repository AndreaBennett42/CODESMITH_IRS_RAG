package gov.irs.directfile.api.io.documentstore;

import gov.irs.directfile.api.io.IOLocationException;

public class DocumentStoreException extends IOLocationException {
    public DocumentStoreException(String message) {
        super(message);
    }

    public DocumentStoreException(String message, Throwable cause) {
        super(message, cause);
    }
}
