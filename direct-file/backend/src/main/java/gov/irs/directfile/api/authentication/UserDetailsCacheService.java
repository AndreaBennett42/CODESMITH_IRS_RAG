package gov.irs.directfile.api.authentication;

import java.util.Optional;
import java.util.UUID;

import org.springframework.security.core.userdetails.UserDetails;

public interface UserDetailsCacheService {
    Optional<UserDetails> get(UUID userExternalId);

    void put(UUID userExternalId, UserDetails userDetails);

    void clear();
}
