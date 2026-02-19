package gov.irs.directfile.api.audit;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.slf4j.spi.LoggingEventBuilder;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.events.Event;
import gov.irs.directfile.api.events.EventStatus;

@Service
@AllArgsConstructor
@Slf4j
public class AuditService {
    AuditEventContextHolder auditEventContextHolder;

    public void addAuditPropertiesToMDC(final Event event) {
        MDC.put(AuditLogElement.EVENT_STATUS.toString(), event.getEventStatus().toString());
        MDC.put(AuditLogElement.EVENT_ID.toString(), event.getEventId().toString());

        if (event.getEventPrincipal().getUserType() != null) {
            MDC.put(
                    AuditLogElement.USER_TYPE.toString(),
                    event.getEventPrincipal().getUserType().toString());
        }
        if (event.getEventErrorMessage() != null) {
            MDC.put(AuditLogElement.EVENT_ERROR_MESSAGE.toString(), event.getEventErrorMessage());
        }
    }

    public void addEventProperty(AuditLogElement property, Object value) {
        if (value != null) {
            auditEventContextHolder.addValueToEventMap(property, value);
        }
    }

    // keeping this package-private, there is currently no need to call it from outside this package
    void performLog() {
        LoggingEventBuilder builder =
                EventStatus.SUCCESS.toString().equals(MDC.get(AuditLogElement.EVENT_STATUS.toString()))
                        ? log.atInfo()
                        : log.atError();
        // Add event-specific elements
        auditEventContextHolder.getEventContextProperties().forEach(builder::addKeyValue);

        builder.addKeyValue(AuditLogElement.CYBER_ONLY.toString(), true).log();
    }
}
