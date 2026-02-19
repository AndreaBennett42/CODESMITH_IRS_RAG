package gov.irs.directfile.api.user;

import java.util.Optional;
import java.util.UUID;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.audit.AuditService;
import gov.irs.directfile.api.config.identity.IdentityAttributes;
import gov.irs.directfile.api.config.identity.IdentitySupplier;
import gov.irs.directfile.api.user.domain.UserInfo;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.audit.events.TinType;

@Service
public class UserService {
    private final UserRepository userRepo;
    private final IdentitySupplier identitySupplier;
    private final AuditService auditService;

    public UserService(
            final UserRepository userRepo, final IdentitySupplier identitySupplier, final AuditService auditService) {
        this.userRepo = userRepo;
        this.identitySupplier = identitySupplier;
        this.auditService = auditService;
    }

    public UserInfo getCurrentUserInfo() {
        IdentityAttributes attributes = identitySupplier.get();
        auditService.addEventProperty(AuditLogElement.USER_TIN, attributes.tin());
        auditService.addEventProperty(AuditLogElement.USER_TIN_TYPE, TinType.INDIVIDUAL.toString());
        return new UserInfo(attributes.id(), attributes.externalId(), attributes.email(), attributes.tin());
    }

    @Transactional(readOnly = true)
    public Optional<User> getUser(UUID userId) {
        return userRepo.findById(userId);
    }

    @Transactional(readOnly = true)
    public Optional<User> getOrCreateUserDev() {
        UUID externalId = UUID.fromString("00000000-0000-0000-0000-000000000000");
        Optional<User> optUser;
        optUser = userRepo.findById(UUID.fromString("11111111-1111-1111-1111-111111111111"));
        if (optUser.isEmpty()) {
            optUser = userRepo.findByExternalId(externalId);
        }
        if (optUser.isEmpty()) {
            User u = new User(externalId);
            u.setAccessGranted(true);
            return Optional.of(userRepo.save(u));
        }
        return optUser;
    }
}
