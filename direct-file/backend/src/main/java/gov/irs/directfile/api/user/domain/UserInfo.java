package gov.irs.directfile.api.user.domain;

import java.util.UUID;

public record UserInfo(UUID id, UUID externalId, String email, String tin) {}
