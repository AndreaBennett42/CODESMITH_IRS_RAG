package gov.irs.directfile.api.events;

import lombok.Getter;

@Getter
public class EventPrincipal {
    private final String userId;
    private final String email;
    private final UserType userType;

    public EventPrincipal() {
        this(null, null, null);
    }

    public EventPrincipal(String userId, String email, UserType userType) {
        this.userId = userId;
        this.email = email;
        this.userType = userType;
    }
}
