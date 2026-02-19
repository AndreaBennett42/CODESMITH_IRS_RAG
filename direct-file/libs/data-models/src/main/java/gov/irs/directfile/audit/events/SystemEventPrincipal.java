package gov.irs.directfile.audit.events;

public class SystemEventPrincipal extends EventPrincipal {
    public SystemEventPrincipal() {
        super(null, UserType.SYS);
    }
}
