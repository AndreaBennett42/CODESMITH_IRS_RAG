package gov.irs.directfile.api.user;

import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.audit.AuditService;
import gov.irs.directfile.api.config.identity.IdentityAttributes;
import gov.irs.directfile.api.config.identity.IdentitySupplier;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.audit.events.TinType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    static final String TIN = "111223333";

    @Mock
    UserRepository userRepository;

    @Mock
    IdentitySupplier identitySupplier;

    @Mock
    AuditService auditService;

    @InjectMocks
    UserService userService;

    @Test
    void givenUserExists_whenSearchingById_thenShouldBeFound() {
        // given
        User testUser1 = new User(UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161"));
        when(userRepository.findById(testUser1.getExternalId())).thenReturn(Optional.of(testUser1));

        // when
        Optional<User> optionalUser = userService.getUser(testUser1.getExternalId());

        // then
        assertTrue(optionalUser.isPresent());
        User found1 = optionalUser.get();
        assertThat(found1).isEqualTo(testUser1);

        // and given
        User testUser2 = new User(UUID.fromString("3274a0db-7465-4e49-aa27-14472c34c9d7"));
        when(userRepository.findById(testUser2.getExternalId())).thenReturn(Optional.of(testUser2));

        // when
        optionalUser = userService.getUser(testUser2.getExternalId());

        // then
        assertTrue(optionalUser.isPresent());
        User found2 = optionalUser.get();
        assertThat(found2).isEqualTo(testUser2);
    }

    @Test
    public void givenUserDoesNotExist_whenSearchingById_thenShouldNotBeFound() {
        // given
        UUID invalidId = UUID.randomUUID();

        // when
        Optional<User> found = userService.getUser(invalidId);

        // then
        assertTrue(found.isEmpty());
    }

    @Test
    public void test() {
        // given
        when(identitySupplier.get())
                .thenReturn(
                        new IdentityAttributes(UUID.randomUUID(), UUID.randomUUID(), "userservicetest@email.com", TIN));

        // when
        userService.getCurrentUserInfo();

        // then
        verify(auditService, times(1)).addEventProperty(AuditLogElement.USER_TIN, TIN);
        verify(auditService, times(1)).addEventProperty(AuditLogElement.USER_TIN_TYPE, TinType.INDIVIDUAL.toString());
    }
}
