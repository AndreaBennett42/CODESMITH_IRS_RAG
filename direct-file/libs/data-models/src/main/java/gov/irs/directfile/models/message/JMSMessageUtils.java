package gov.irs.directfile.models.message;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import jakarta.jms.JMSException;
import jakarta.jms.Message;

/**
 * This utility class facilitates working with both SQSMessage and JMSMessage.
 * SQSMessage provides access to all attributes as a map, while JMSMessage does
 * not support retrieving all attributes in a similar manner. This class helps
 * bridge that gap.
 */
public class JMSMessageUtils {

    /**
     * Retrieves all properties from a JMS message and returns them as a Map.
     *
     * @param message The JMS message from which to retrieve properties.
     * @return A Map containing all properties of the JMS message.
     * @throws JMSException If an error occurs while accessing the message
     *                      properties.
     */
    public static Map<String, String> getMessagePropertiesAsMap(Message message) throws JMSException {
        Map<String, String> propertiesMap = new HashMap<>();

        // Get all the property names and their values from the JMS message
        for (@SuppressWarnings("unchecked") Enumeration<String> propertyNames = message.getPropertyNames();
                propertyNames.hasMoreElements(); ) {
            String propertyName = propertyNames.nextElement();
            String propertyValue = message.getStringProperty(propertyName);
            propertiesMap.put(propertyName, propertyValue);
        }

        return propertiesMap;
    }
}
