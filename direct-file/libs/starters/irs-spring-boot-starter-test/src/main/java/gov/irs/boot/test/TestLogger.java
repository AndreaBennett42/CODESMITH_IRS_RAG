package gov.irs.boot.test;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import jakarta.annotation.Nonnull;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.slf4j.LoggerFactory;

@RequiredArgsConstructor
public class TestLogger implements BeforeEachCallback, AfterEachCallback {

    private final Level level;
    private final String loggerName;

    @Getter
    private Logger logger;

    @Getter
    private ListAppender<ILoggingEvent> appender;

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

    public int errorCount() {
        return countEvents(Level.ERROR);
    }

    public int infoCount() {
        return countEvents(Level.INFO);
    }

    public int countEvents(@Nonnull Level level) {
        int count = 0;

        for (ILoggingEvent loggedEvent : appender.list) {
            if (loggedEvent.getLevel().equals(level)) {
                count++;
            }
        }

        return count;
    }

    /**
     * Returns true if it finds an {@link ILoggingEvent} with the same level and message as the parameter event
     * @param event
     * @return
     */
    public boolean hasEvent(ILoggingEvent event) {
        return hasEvent(event.getLevel(), event.getMessage());
    }

    public boolean hasEvent(Level level, String message) {
        for (ILoggingEvent loggedEvent : appender.list) {
            if (loggedEvent.getLevel().equals(level) && loggedEvent.getMessage().equals(message)) {
                return true;
            }
        }

        return false;
    }
}
