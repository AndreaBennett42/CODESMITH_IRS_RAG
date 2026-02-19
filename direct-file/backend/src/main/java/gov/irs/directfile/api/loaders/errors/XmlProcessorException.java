package gov.irs.directfile.api.loaders.errors;

public class XmlProcessorException extends RuntimeException {
    public XmlProcessorException(final String message) {
        super(message);
    }

    public XmlProcessorException(final String message, Throwable cause) {
        super(message, cause);
    }
}
