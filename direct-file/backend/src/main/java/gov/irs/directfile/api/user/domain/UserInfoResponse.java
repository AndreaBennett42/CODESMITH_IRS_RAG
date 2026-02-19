package gov.irs.directfile.api.user.domain;

public record UserInfoResponse(String email) {
    public UserInfoResponse(UserInfo userInfo) {
        this(userInfo.email() != null ? userInfo.email() : "");
    }
}
