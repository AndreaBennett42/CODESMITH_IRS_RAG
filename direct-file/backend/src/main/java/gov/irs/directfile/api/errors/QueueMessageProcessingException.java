package gov.irs.directfile.api.errors;

public class QueueMessageProcessingException extends RuntimeException {

    public QueueMessageProcessingException() {
        super();
    }

    public QueueMessageProcessingException(String message) {
        super(message);
    }
}
