package gov.irs.directfile.stateapi.audit;

import java.util.List;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.netty.handler.codec.http.HttpResponseStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;

import gov.irs.directfile.stateapi.events.Event;
import gov.irs.directfile.stateapi.events.EventId;
import gov.irs.directfile.stateapi.events.EventStatus;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ExtendWith(MockitoExtension.class)
public class AuditServiceTest {

    private AuditService auditSvc = new AuditService();
    private Logger auditLogger;
    private ListAppender<ILoggingEvent> listAppender;

    @BeforeEach
    public void setup() {

        auditLogger = (Logger) LoggerFactory.getLogger(AuditService.class);
        listAppender = new ListAppender<>();
        listAppender.setContext((LoggerContext) LoggerFactory.getILoggerFactory());
        listAppender.start();
        auditLogger.addAppender(listAppender);
    }

    @AfterEach
    public void tearDown() {
        auditLogger.detachAppender(listAppender);
    }

    @Test
    public void testLogEvent_success() throws Exception {

        Event event = createAuditEvent(
                EventId.CREATE_AUTHORIZATION_CODE, EventStatus.SUCCESS, HttpResponseStatus.OK.toString(), null);
        auditSvc.logEvent(event);

        List<ILoggingEvent> logsList = listAppender.list;
        assertEquals(Level.INFO, logsList.get(0).getLevel());
        assertEquals(logsList.get(0).getMDCPropertyMap().get("eventId"), EventId.CREATE_AUTHORIZATION_CODE.toString());
        assertEquals(logsList.get(0).getMDCPropertyMap().get("eventStatus"), EventStatus.SUCCESS.toString());
        assertEquals(logsList.get(0).getMDCPropertyMap().get("responseStatusCode"), HttpResponseStatus.OK.toString());
    }

    @Test
    public void testLogEvent_failure() throws Exception {

        String errMsg = "Internal Server Error";
        Event event = createAuditEvent(
                EventId.CREATE_AUTHORIZATION_CODE,
                EventStatus.FAILURE,
                HttpResponseStatus.INTERNAL_SERVER_ERROR.toString(),
                errMsg);
        auditSvc.logEvent(event);

        List<ILoggingEvent> logsList = listAppender.list;
        assertEquals(Level.ERROR, logsList.get(0).getLevel());
        assertEquals(logsList.get(0).getMDCPropertyMap().get("eventId"), EventId.CREATE_AUTHORIZATION_CODE.toString());
        assertEquals(logsList.get(0).getMDCPropertyMap().get("eventStatus"), EventStatus.FAILURE.toString());
        assertEquals(
                logsList.get(0).getMDCPropertyMap().get("responseStatusCode"),
                HttpResponseStatus.INTERNAL_SERVER_ERROR.toString());
        assertEquals(logsList.get(0).getMDCPropertyMap().get("eventErrorMessage"), errMsg);
    }

    private Event createAuditEvent(EventId id, EventStatus eventStatus, String responseStatus, String errorMsg) {

        Event event = Event.builder()
                .eventId(id)
                .eventStatus(eventStatus)
                .responseStatusCode(responseStatus)
                .eventErrorMessage(errorMsg)
                .taxPeriod("2022")
                .userType("SYS")
                .stateId("FS")
                .remoteAddress("127.0.0.1")
                .build();
        return event;
    }
}
