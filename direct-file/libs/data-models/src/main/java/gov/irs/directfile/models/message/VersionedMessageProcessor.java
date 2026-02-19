package gov.irs.directfile.models.message;

import java.util.Map;

/**
 * Interface for processing messages with versioning support.
 * <p>
 * Implementations of this interface are responsible for processing messages
 * that include a unique identifier, associated attributes, and the message
 * body.
 */
public interface VersionedMessageProcessor {
    /**
     * Processes a message based on its ID, attributes, and body content. This
     * method is responsible for handling the processing logic for a message with
     * the specified version. The method takes the message ID, a map of attributes,
     * and the message body as parameters.
     *
     * Unlike SQSMessage (e.g. used in Data-Import Lambda), JMSMessage does
     * not support retrieving all attributes as a map. {@link JMSMessageUtils}
     * provides method to get all attributes as a map.
     *
     * <p>
     * Example usage:
     * </p>
     *
     * <pre>
     * private void extractAndSaveUserData(Message message) throws JMSException {
     *     VersionedMessageProcessor handler = messageProcessors.get(version);
     *     if (handler != null) {
     *         handler.process(message.getJMSMessageID(),
     *                 JMSMessageUtils.getMessagePropertiesAsMap(message),
     *                 ((TextMessage) message).getText());
     *     } else {
     *         throw new UnsupportedVersionException("No handler found for version " + version);
     *     }
     * }
     * </pre>
     *
     * @param messageId  the unique identifier of the message to be processed
     * @param attributes a map of key-value pairs representing the attributes of the
     *                   message
     * @param body       the actual content of the message that needs to be
     *                   processed
     *
     * @throws RuntimeException if the processing of the message fails for any
     *                          reason.
     */
    void process(String messageId, Map<String, String> attributes, String body);

    /**
     * Retrieves the version identifier associated with this message processor.
     *
     * This method returns a string representing the version of the message format
     * that this processor is capable of handling. The version identifiers are
     * typically constants defined in the {@link MessageVersion} class, such as
     * "1.0", "2.0", etc.
     *
     * The returned version string helps in routing messages to the appropriate
     * processor based on the message's version.
     *
     * @return the version identifier as a string, indicating the message format
     *         version this processor handles
     */
    String version();
}
