package gov.irs.directfile.api.audit;

import java.util.HashMap;
import java.util.Map;

import org.springframework.stereotype.Component;
import org.springframework.web.context.annotation.RequestScope;

@Component
@RequestScope
public class AuditEventContextHolder {
    private final Map<String, Object> auditEventProperties = new HashMap<>();
    private final Map<String, Object> auditEventDetailProperties = new HashMap<>();

    public Map<String, Object> getEventContextProperties() {
        Map<String, Object> outputProperties = new HashMap<>(auditEventProperties);
        if (!auditEventDetailProperties.isEmpty()) {
            outputProperties.put(AuditLogElement.DETAIL.toString(), auditEventDetailProperties);
        }
        return outputProperties;
    }

    public void addValueToEventMap(AuditLogElement key, Object value) {
        auditEventProperties.put(key.toString(), value);
    }

    public void addValueToEventDetailMap(AuditLogElement.DetailElement key, String value) {
        auditEventDetailProperties.put(key.toString(), value);
    }
}
