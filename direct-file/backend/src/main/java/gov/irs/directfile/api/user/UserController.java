package gov.irs.directfile.api.user;

import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.RestController;

import gov.irs.directfile.api.audit.Auditable;
import gov.irs.directfile.api.events.EventId;
import gov.irs.directfile.api.user.domain.UserInfoResponse;

@RestController
@AllArgsConstructor
public class UserController implements UserApi {
    private final UserService userService;

    @Override
    @Auditable(event = EventId.USER_INFO_GET)
    public UserInfoResponse getUserInfo() {
        return new UserInfoResponse(userService.getCurrentUserInfo());
    }
}
