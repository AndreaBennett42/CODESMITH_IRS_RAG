package gov.irs.directfile.stateapi.repository.facts;

import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.model.GetStateExportedFactsResponse;

public interface ExportedFactsClient {

    Mono<GetStateExportedFactsResponse> getExportedFacts(String submissionId, String stateCode, String accountId);
}
