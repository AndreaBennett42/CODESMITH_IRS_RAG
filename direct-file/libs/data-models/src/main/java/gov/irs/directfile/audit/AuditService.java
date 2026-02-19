package gov.irs.directfile.audit;

import java.util.Map;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.slf4j.spi.LoggingEventBuilder;
import org.springframework.stereotype.Service;

import gov.irs.directfile.audit.events.Event;
import gov.irs.directfile.audit.events.EventStatus;

@Service
@Slf4j
@SuppressFBWarnings(value = "RV_RETURN_VALUE_IGNORED", justification = "Initial Spotbugs Setup")
public class AuditService {

    public void addAuditPropertiesToMDC(final Event event) {
        MDC.put(AuditLogElement.eventStatus.toString(), event.getEventStatus().toString());
        MDC.put(AuditLogElement.eventId.toString(), event.getEventId().toString());

        if (event.getEventPrincipal().getUserType() != null) {
            MDC.put(
                    AuditLogElement.userType.toString(),
                    event.getEventPrincipal().getUserType().toString());
        }
        if (event.getEventErrorMessage() != null) {
            MDC.put(AuditLogElement.eventErrorMessage.toString(), event.getEventErrorMessage());
        }
    }

    // For logging performed through Aop and request filters
    public void performLog() {
        LoggingEventBuilder builder =
                EventStatus.SUCCESS.toString().equals(MDC.get(AuditLogElement.eventStatus.toString()))
                        ? log.atInfo()
                        : log.atError();

        // Add `cyberOnly`: the property that indicates event should be forwarded to XXXX
        builder.addKeyValue(AuditLogElement.cyberOnly.toString(), true).log();
    }

    // For logging performed directly by populating event data (for now, submit and status apps are this way)
    public void performLogFromEvent(final Event event, final AuditEventData eventData) {
        log(event, eventData);
        MDC.clear();
    }

    public void performLogFromEventAndPreserveMDCValues(final Event event, final AuditEventData eventData) {
        Map<String, String> mdcContext = MDC.getCopyOfContextMap();
        log(event, eventData);
        MDC.setContextMap(mdcContext);
    }

    private void log(Event event, AuditEventData eventData) {
        addAuditPropertiesToMDC(event);

        LoggingEventBuilder builder =
                EventStatus.SUCCESS.toString().equals(MDC.get(AuditLogElement.eventStatus.toString()))
                        ? log.atInfo()
                        : log.atError();

        eventData.getEventData().forEach((k, v) -> builder.addKeyValue(k.toString(), v));

        // Add `cyberOnly`: the property that indicates event should be forwarded to XXXX
        builder.addKeyValue(AuditLogElement.cyberOnly.toString(), true).log();
    }
}
