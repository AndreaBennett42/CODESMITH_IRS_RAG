package gov.irs.directfile.api.audit;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.audit.events.TinType;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AuditEventContextHolderTest {

    @Test
    public void givenSetFields_whenGetEventContextProperties_thenReturnsMapOfProperties() {
        AuditEventContextHolder auditEventContextHolder = new AuditEventContextHolder();
        auditEventContextHolder.addValueToEventMap(AuditLogElement.USER_TIN, "test-tin");
        auditEventContextHolder.addValueToEventMap(AuditLogElement.USER_TIN_TYPE, TinType.INDIVIDUAL.toString());
        Map<String, String> expectedMap = Map.of(
                "userTin", "test-tin",
                "userTinType", "0");
        Map<String, Object> eventsMap = auditEventContextHolder.getEventContextProperties();
        assertEquals(expectedMap, eventsMap);
    }

    @Test
    public void givenEmptyProperties_whenGetEventContextProperties_thenReturnsMapExcludingEmptyValues() {
        AuditEventContextHolder auditEventContextHolder = new AuditEventContextHolder();
        Map<String, Object> eventsMap = auditEventContextHolder.getEventContextProperties();
        assertEquals(new HashMap<>(), eventsMap);
    }
}
