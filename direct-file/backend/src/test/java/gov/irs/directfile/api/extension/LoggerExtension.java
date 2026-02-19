package gov.irs.directfile.api.extension;

import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.slf4j.LoggerFactory;
import org.slf4j.event.KeyValuePair;

import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.events.Event;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class LoggerExtension implements BeforeEachCallback, AfterEachCallback {
    private Logger logger;
    private ListAppender<ILoggingEvent> appender;
    private final String loggerName;
    private final Level level;

    public LoggerExtension(Level level, String loggerName) {
        this.loggerName = loggerName;
        this.level = level;
    }

    @Override
    public void beforeEach(ExtensionContext context) {
        logger = (Logger) LoggerFactory.getLogger(loggerName);
        appender = new ListAppender<ILoggingEvent>();
        appender.setContext((LoggerContext) LoggerFactory.getILoggerFactory());
        logger.setLevel(level);
        logger.addAppender(appender);
        appender.start();
    }

    @Override
    public void afterEach(ExtensionContext context) {
        logger.detachAppender(appender);
    }

    public void verifyLogEvent(Event event) {
        verifyLogEvent(event, 0, null);
    }

    public void verifyLogEvent(Event event, Map<AuditLogElement, Object> logPropertiesToAssert) {
        verifyLogEvent(event, 0, logPropertiesToAssert);
    }

    public void verifyLogEvent(Event event, int index) {
        verifyLogEvent(event, index, null);
    }

    public void verifyLogEvent(Event event, int index, Map<AuditLogElement, Object> logPropertiesToAssert) {
        ILoggingEvent loggingEvent = appender.list.get(index);

        List<KeyValuePair> keyValuePairs = loggingEvent.getKeyValuePairs();
        Map<String, String> mdcPropertyMap = loggingEvent.getMDCPropertyMap();

        HashMap<String, String> combinedMap = new HashMap<>(mdcPropertyMap);
        ListIterator<KeyValuePair> keyValuePairListIterator = keyValuePairs.listIterator();
        while (keyValuePairListIterator.hasNext()) {
            KeyValuePair next = keyValuePairListIterator.next();
            // a collision here causes logback to silently exclude all keyValuePairs from json output
            assertFalse(combinedMap.containsKey(next.key));
            combinedMap.put(next.key, next.value == null ? null : next.value.toString());
        }

        assertNotNull(combinedMap.get(AuditLogElement.REMOTE_ADDRESS.toString()), "Remote address");
        assertNotNull(combinedMap.get(AuditLogElement.REQUEST_METHOD.toString()), "Request method");
        assertNotNull(combinedMap.get(AuditLogElement.REQUEST_URI.toString()), "Request uri");
        assertNotNull(combinedMap.get(AuditLogElement.RESPONSE_STATUS_CODE.toString()), "Response status code");
        assertEquals("true", combinedMap.get(AuditLogElement.CYBER_ONLY.toString()), "cyberOnly not set (XXXX flag)");

        assertEquals(event.getEventStatus().toString(), combinedMap.get(AuditLogElement.EVENT_STATUS.toString()));
        assertEquals(event.getEventId().toString(), combinedMap.get(AuditLogElement.EVENT_ID.toString()));
        assertEquals(event.getEventPrincipal().getUserId(), combinedMap.get(AuditLogElement.SADI_USER_UUID.toString()));
        assertEquals(event.getEventPrincipal().getEmail(), combinedMap.get(AuditLogElement.EMAIL.toString()));
        assertEquals(
                event.getEventPrincipal().getUserType().toString(),
                combinedMap.get(AuditLogElement.USER_TYPE.toString()));

        if (event.getEventErrorMessage() != null) {
            // Only assert on error message if the errorMessage field is provided?
            assertEquals(event.getEventErrorMessage(), combinedMap.get(AuditLogElement.EVENT_ERROR_MESSAGE.toString()));
        }

        assertEquals(event.getTaxPeriod(), combinedMap.get(AuditLogElement.TAX_PERIOD.toString()));
        assertEquals(event.getUserTin(), combinedMap.get(AuditLogElement.USER_TIN.toString()));
        assertEquals(event.getUserTinType(), combinedMap.get(AuditLogElement.USER_TIN_TYPE.toString()));

        if (logPropertiesToAssert != null) {
            for (Map.Entry<AuditLogElement, Object> entry : logPropertiesToAssert.entrySet()) {
                assertEquals(
                        entry.getValue().toString(),
                        combinedMap.get(entry.getKey().toString()));
            }
        }
    }
}
