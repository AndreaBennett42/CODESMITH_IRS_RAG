package gov.irs.directfile.stateapi.repository;

import java.util.UUID;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.configuration.DirectFileEndpointProperties;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.exception.StateApiExportedFactsDisabledException;
import gov.irs.directfile.stateapi.model.GetStateExportedFactsResponse;
import gov.irs.directfile.stateapi.model.TaxReturnStatus;

@Service
@Slf4j
public class DirectFileBackendService {

    private final WebClient webClient;
    private final String baseUri;

    public DirectFileBackendService(DirectFileEndpointProperties dfEndpointProperties) {
        webClient = WebClient.builder()
                .defaultStatusHandler(
                        HttpStatusCode::isError, resp -> resp.createException().flatMap(e -> {
                            log.error(
                                    "DirectFileBackendService failed, {}, error: {}",
                                    e.getClass().getName(),
                                    e.getMessage());
                            return Mono.just(new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR));
                        }))
                .baseUrl(dfEndpointProperties.getBackendUrl())
                .build();

        baseUri = dfEndpointProperties.getBackendContextPath() + "/" + dfEndpointProperties.getBackendApiVersion();
    }

    public Mono<GetStateExportedFactsResponse> getExportedFacts(
            String submissionId, String stateCode, String accountId) {
        String efUri = baseUri + "/state-api/state-exported-facts/{submissionId}";
        return webClient
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path(efUri)
                        .queryParam("stateCode", stateCode)
                        .queryParam("accountId", accountId)
                        .build(submissionId))
                .retrieve()
                .onStatus(
                        HttpStatus.METHOD_NOT_ALLOWED::equals,
                        clientResponse -> Mono.error(new StateApiExportedFactsDisabledException()))
                .bodyToMono(GetStateExportedFactsResponse.class);
    }

    public Mono<TaxReturnStatus> getStatus(int taxYear, UUID taxReturnId, String submissionId) {
        String statusUri = baseUri + "/state-api/status/{taxFilingYear}/{taxReturnId}/{submissionId}";
        return webClient
                .get()
                .uri(uriBuilder -> uriBuilder.path(statusUri).build(taxYear, taxReturnId, submissionId))
                .retrieve()
                .bodyToMono(TaxReturnStatus.class);
    }
}
