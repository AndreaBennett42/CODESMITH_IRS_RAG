package gov.irs.directfile.api.user;

import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.api.user.models.User;

public interface UserRepository extends CrudRepository<User, UUID> {
    @Query
    Optional<User> findByExternalId(final UUID externalId);

    @Query
    int countByAccessGranted(boolean accessGranted);
}
