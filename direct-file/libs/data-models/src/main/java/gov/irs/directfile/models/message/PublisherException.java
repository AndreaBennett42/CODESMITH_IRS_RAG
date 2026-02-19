package gov.irs.directfile.models.message;

public class PublisherException extends RuntimeException {
    public PublisherException(String message) {
        super(message);
    }

    public PublisherException(String message, Throwable cause) {
        super(message, cause);
    }
}
