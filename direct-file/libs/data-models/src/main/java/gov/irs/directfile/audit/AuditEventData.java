package gov.irs.directfile.audit;

import java.util.HashMap;
import java.util.Map;

import lombok.Getter;

public class AuditEventData {
    @Getter
    private final Map<AuditLogElement, Object> eventData = new HashMap<>();

    private final Map<String, Object> eventDetailData = new HashMap<>();

    public AuditEventData() {
        eventData.put(AuditLogElement.detail, eventDetailData);
    }

    public void put(AuditLogElement key, Object value) {
        eventData.put(key, value);
    }

    public void putDetail(String key, Object value) {
        eventDetailData.put(key, value);
    }
}
