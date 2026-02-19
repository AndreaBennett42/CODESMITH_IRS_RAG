package gov.irs.directfile.stateapi.service;

import java.util.Map;
import java.util.UUID;

import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.dto.StateProfileDTO;
import gov.irs.directfile.stateapi.model.*;

public interface StateApiService {

    Mono<UUID> createAuthorizationCode(AuthCodeRequest ac);

    Mono<String> generateAuthorizationToken(AuthCodeRequest ac);

    Mono<StateAndAuthCode> verifyJwtSignature(String jwt, String accountId);

    Mono<AuthorizationCode> authorize(StateAndAuthCode stateAndAuthCode);

    Mono<TaxReturnXml> retrieveTaxReturnXml(int taxYear, UUID taxReturnUuid, String submissionId);

    Mono<EncryptData> encryptTaxReturn(TaxReturnToExport taxReturn, String accountId);

    Mono<StateProfileDTO> lookupStateProfile(String stateCode);

    Mono<StateProfile> getStateProfile(String accountId);

    Mono<Map<String, Object>> retrieveExportedFacts(String submissionId, String stateCode, String accountId);
}
