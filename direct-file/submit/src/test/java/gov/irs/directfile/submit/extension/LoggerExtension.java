package gov.irs.directfile.submit.extension;

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

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.audit.events.Event;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

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
        appender = new ListAppender<>();
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
            // a collision here causes logback to silently exclude all keyValuePairs from
            // json output
            assertFalse(combinedMap.containsKey(next.key));
            combinedMap.put(next.key, next.value == null ? null : next.value.toString());
        }

        assertEquals(combinedMap.get(AuditLogElement.cyberOnly.toString()), "true", "cyberOnly not set (XXXX flag)");
        assertEquals(event.getEventStatus().toString(), combinedMap.get(AuditLogElement.eventStatus.toString()));
        assertEquals(event.getEventId().toString(), combinedMap.get(AuditLogElement.eventId.toString()));
        assertEquals(
                event.getEventPrincipal().getUserType().toString(),
                combinedMap.get(AuditLogElement.userType.toString()));
        assertEquals(event.getEventErrorMessage(), combinedMap.get(AuditLogElement.eventErrorMessage.toString()));
        assertEquals(event.getDetail(), combinedMap.get(AuditLogElement.detail.toString()));

        if (logPropertiesToAssert != null) {
            for (Map.Entry<AuditLogElement, Object> entry : logPropertiesToAssert.entrySet()) {
                assertEquals(
                        entry.getValue().toString(),
                        combinedMap.get(entry.getKey().toString()));
            }
        }
    }
}
