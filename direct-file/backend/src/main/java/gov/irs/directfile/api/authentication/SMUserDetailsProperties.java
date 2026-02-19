package gov.irs.directfile.api.authentication;

import java.util.UUID;

public record SMUserDetailsProperties(UUID id, UUID externalId, String email, String tin) {
    public SMUserDetailsProperties(SMUserDetailsPrincipal principal) {
        this(principal.id(), principal.externalId(), principal.email(), principal.tin());
    }
}
