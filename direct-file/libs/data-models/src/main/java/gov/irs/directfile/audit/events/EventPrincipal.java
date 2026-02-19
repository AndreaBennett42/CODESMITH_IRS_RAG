package gov.irs.directfile.audit.events;

import lombok.Getter;

@Getter
public class EventPrincipal {
    private final String userId;
    private final UserType userType;

    public EventPrincipal() {
        this(null, null);
    }

    public EventPrincipal(String userId, UserType userType) {
        this.userId = userId;
        this.userType = userType;
    }
}
