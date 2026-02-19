package gov.irs.directfile.api.events;

import org.slf4j.MDC;

import gov.irs.directfile.api.audit.AuditLogElement;

public class TaxpayerEventPrincipal extends EventPrincipal {

    public TaxpayerEventPrincipal() {
        super();
    }

    public TaxpayerEventPrincipal(String userId, String email) {
        super(userId, email, UserType.SYS);
    }

    public static TaxpayerEventPrincipal createFromContext() {
        String userId = MDC.get(AuditLogElement.SADI_USER_UUID.toString());
        String email = MDC.get(AuditLogElement.EMAIL.toString());
        return new TaxpayerEventPrincipal(userId, email);
    }
}
