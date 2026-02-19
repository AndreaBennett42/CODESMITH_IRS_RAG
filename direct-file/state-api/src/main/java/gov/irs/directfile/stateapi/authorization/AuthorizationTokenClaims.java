package gov.irs.directfile.stateapi.authorization;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record AuthorizationTokenClaims(String taxReturnUuid, String submissionId, String stateCode, int taxYear) {}
