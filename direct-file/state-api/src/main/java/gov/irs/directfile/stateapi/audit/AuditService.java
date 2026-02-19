package gov.irs.directfile.stateapi.audit;

import java.util.Map;

import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.slf4j.spi.LoggingEventBuilder;
import org.springframework.stereotype.Service;

import gov.irs.directfile.stateapi.events.Event;
import gov.irs.directfile.stateapi.events.EventStatus;

@Service
@Slf4j
public class AuditService {

    public void logEvent(Event event) {

        if (event != null) {
            Map<String, String> contextMapCopy = MDC.getCopyOfContextMap();
            LoggingEventBuilder loggingEventBuilder =
                    (event.getEventStatus() == EventStatus.FAILURE) ? log.atError() : log.atInfo();

            try {
                if (event.getEventId() != null) {
                    MDC.put(
                            AuditLogElement.eventId.toString(),
                            event.getEventId().toString());
                }
                if (event.getEventStatus() != null) {
                    MDC.put(
                            AuditLogElement.eventStatus.toString(),
                            event.getEventStatus().toString());
                }
                if (event.getResponseStatusCode() != null) {
                    MDC.put(AuditLogElement.responseStatusCode.toString(), event.getResponseStatusCode());
                }
                if (event.getEventErrorMessage() != null) {
                    MDC.put(AuditLogElement.eventErrorMessage.toString(), event.getEventErrorMessage());
                }
                if (event.getTaxPeriod() != null) {
                    MDC.put(AuditLogElement.taxPeriod.toString(), event.getTaxPeriod());
                }
                if (event.getStateId() != null) {
                    MDC.put(AuditLogElement.stateId.toString(), event.getStateId());
                }
                if (event.getUserType() != null) {
                    MDC.put(AuditLogElement.userType.toString(), event.getUserType());
                }
                if (event.getTaxReturnId() != null) {
                    MDC.put(AuditLogElement.taxReturnId.toString(), event.getTaxReturnId());
                }
                if (event.getDetail() != null) {
                    loggingEventBuilder = loggingEventBuilder.addKeyValue(
                            AuditLogElement.detail.toString(), event.getDetail().getDetailMap());
                }

                MDC.put(AuditLogElement.eventType.toString(), "STATE_API");
                MDC.put(AuditLogElement.cyberOnly.toString(), Boolean.TRUE.toString());

                loggingEventBuilder.log();
            } finally {
                // reset the MDC context map
                MDC.setContextMap(contextMapCopy);
            }
        }
    }
}
