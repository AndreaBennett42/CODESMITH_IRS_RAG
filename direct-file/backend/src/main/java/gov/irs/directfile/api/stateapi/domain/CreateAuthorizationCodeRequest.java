package gov.irs.directfile.api.stateapi.domain;

import java.util.UUID;

import jakarta.validation.constraints.NotNull;

public record CreateAuthorizationCodeRequest(@NotNull UUID taxReturnUuid, @NotNull int taxYear) {}
