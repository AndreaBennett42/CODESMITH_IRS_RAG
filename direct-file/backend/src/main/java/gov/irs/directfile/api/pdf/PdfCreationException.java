package gov.irs.directfile.api.pdf;

public class PdfCreationException extends Exception {
    public PdfCreationException(String message) {
        super(message);
    }

    public PdfCreationException(Throwable e) {
        super(e);
    }

    public PdfCreationException(String message, Throwable cause) {
        super(message, cause);
    }
}
