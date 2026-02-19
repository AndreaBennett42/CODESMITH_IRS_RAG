package gov.irs.directfile.models.message;

public interface QueueMessage<T> {
    T getPayload();

    QueueMessageHeaders getHeaders();
}
