package gov.irs.directfile.audit;

import java.util.Map;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import org.slf4j.Logger;
import org.slf4j.spi.LoggingEventBuilder;

/*
 * This is for non-audit log messages where we cannot use the slf4j MDC
 * but also want to have more structure
 * like encryption of some fields.
 */

@SuppressFBWarnings(value = "RV_RETURN_VALUE_IGNORED", justification = "Initial Spotbugs Setup")
public class LogUtil {
    private final Logger log;

    public LogUtil(Logger log) {
        // using logger from other source class will keep `logger_name` correct in messages instead of showing LogUtil
        this.log = log;
    }

    public void logAtInfo(String message, Map<AuditLogElement, Object> properties) {
        LoggingEventBuilder builder = log.atInfo();
        performLog(builder, message, properties);
    }

    public void logAtError(String message, Map<AuditLogElement, Object> properties, Throwable throwable) {
        LoggingEventBuilder builder = log.atError().setCause(throwable);
        performLog(builder, message, properties);
    }

    private void performLog(LoggingEventBuilder builder, String message, Map<AuditLogElement, Object> properties) {
        builder.setMessage(message);
        properties.forEach((k, v) -> builder.addKeyValue(k.toString(), v.toString()));

        builder.addKeyValue(AuditLogElement.cyberOnly.toString(), true).log();
    }
}
