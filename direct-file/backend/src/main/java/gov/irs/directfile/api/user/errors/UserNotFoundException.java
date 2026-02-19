package gov.irs.directfile.api.user.errors;

import java.util.UUID;

public class UserNotFoundException extends RuntimeException {
    public UserNotFoundException(final UUID userId) {
        super(String.format("User %s not found", userId));
    }
}
