package gov.irs.directfile.api.stateapi.domain;

import java.util.UUID;

import jakarta.validation.constraints.NotNull;

import gov.irs.directfile.models.StateOrProvince;

public record StateApiCreateAuthorizationCodeRequest(
        @NotNull UUID taxReturnUuid,
        String tin,
        @NotNull int taxYear,
        @NotNull StateOrProvince stateCode,
        String submissionId) {}
