package gov.irs.directfile.api.user;

import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.api.util.base.BaseRepositoryTest;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class UserRepositoryTest extends BaseRepositoryTest {

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    UserRepository userRepo;

    @Test
    public void givenDetailsAreInvalid_whenSearchingUser_thenShouldNotBeFound() {
        // given
        UUID userUuid = UUID.fromString("97429a5d-7748-4a09-9157-63cff181a6da");

        // when
        Optional<User> result = userRepo.findByExternalId(userUuid);

        // then
        assertTrue(result.isEmpty());
    }

    @Test
    public void givenDetailsAreValid_whenSearchingUser_thenShouldBeFound() {
        // given
        UUID userUuid = UUID.fromString("97429a5d-7748-4a09-9157-63cff181a6da");
        User user = new User(userUuid);
        entityManager.persist(user);

        // when
        Optional<User> result = userRepo.findByExternalId(userUuid);

        // then
        assertTrue(result.isPresent());
        assertEquals(userUuid, result.get().getExternalId());
    }

    @Test
    void givenUser_whenSaved_thenAccessGrantedFieldDefaultsToFalse() {
        // given
        UUID userUuid = UUID.randomUUID();

        // when
        User user = new User(userUuid);
        assertFalse(user.isAccessGranted());

        userRepo.save(user);
        Optional<User> result = userRepo.findByExternalId(userUuid);
        assertTrue(result.isPresent());

        // then
        assertFalse(user.isAccessGranted());
    }

    @Test
    void givenUser_whenAccessGrantedColumnSetToTrue_thenAccessGrantedSavesAsTrue() {
        // given
        UUID userUuid = UUID.randomUUID();

        // when
        User user = new User(userUuid);
        user.setAccessGranted(true);

        userRepo.save(user);
        Optional<User> result = userRepo.findByExternalId(userUuid);
        assertTrue(result.isPresent());

        // then
        assertTrue(user.isAccessGranted());
    }

    @Test
    void givenUserTable_whenCountingByAccessGranted_thenAccurateCountsReturned() {
        assertEquals(0, userRepo.countByAccessGranted(true));

        // given
        UUID userUuid = UUID.randomUUID();
        User user = new User(userUuid);
        user.setAccessGranted(true);
        userRepo.save(user);

        // when
        int result = userRepo.countByAccessGranted(true);

        // then
        assertEquals(1, result);
    }
}
