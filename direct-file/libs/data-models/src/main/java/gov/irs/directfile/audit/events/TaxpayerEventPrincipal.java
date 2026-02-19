package gov.irs.directfile.audit.events;

import org.slf4j.MDC;

import gov.irs.directfile.audit.AuditLogElement;

public class TaxpayerEventPrincipal extends EventPrincipal {

    public TaxpayerEventPrincipal() {
        super();
    }

    public TaxpayerEventPrincipal(String userId) {
        super(userId, UserType.REGT);
    }

    public static TaxpayerEventPrincipal createFromContext() {
        String userId = MDC.get(AuditLogElement.sadiUserUuid.toString());
        return new TaxpayerEventPrincipal(userId);
    }
}
