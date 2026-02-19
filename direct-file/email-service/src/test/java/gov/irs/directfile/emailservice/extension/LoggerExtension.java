package gov.irs.directfile.emailservice.extension;

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

import static org.junit.jupiter.api.Assertions.assertEquals;

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

    public List<ILoggingEvent> getLoggingEvents() {
        return appender.list;
    }

    public void verifyKeyValuePairsEqual(ILoggingEvent event, Map<String, String> expectedKeyValuePairs) {
        ListIterator<KeyValuePair> keyValuePairListIterator =
                event.getKeyValuePairs().listIterator();
        Map<String, String> actualKeyValuePairs = new HashMap<>();

        while (keyValuePairListIterator.hasNext()) {
            KeyValuePair next = keyValuePairListIterator.next();
            actualKeyValuePairs.put(next.key, next.value == null ? null : next.value.toString());
        }

        assertEquals(expectedKeyValuePairs, actualKeyValuePairs);
    }
}
