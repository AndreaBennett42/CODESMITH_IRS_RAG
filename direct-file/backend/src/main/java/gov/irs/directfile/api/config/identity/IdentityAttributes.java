package gov.irs.directfile.api.config.identity;

import java.util.UUID;

public record IdentityAttributes(UUID id, UUID externalId, String email, String tin) {}
