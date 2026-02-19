package gov.irs.directfile.api.user;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import gov.irs.directfile.api.user.domain.UserInfoResponse;

@RequestMapping("${direct-file.api-version}/users")
@Validated
@Tag(name = "users", description = "The user API")
public interface UserApi {
    @Operation(summary = "Get current user info", description = "Find a tax return by its ID")
    @GetMapping("me")
    UserInfoResponse getUserInfo();
}
