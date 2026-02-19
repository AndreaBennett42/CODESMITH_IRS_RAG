package gov.irs.directfile.api.events;

public class SystemEventPrincipal extends EventPrincipal {
    public SystemEventPrincipal() {
        super(null, null, UserType.SYS);
    }
}
