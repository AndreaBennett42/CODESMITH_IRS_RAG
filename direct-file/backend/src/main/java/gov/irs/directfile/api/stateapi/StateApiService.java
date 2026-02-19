package gov.irs.directfile.api.stateapi;

import com.fasterxml.jackson.core.JsonProcessingException;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import gov.irs.factgraph.Graph;

import gov.irs.directfile.api.config.StateApiEndpointProperties;
import gov.irs.directfile.api.loaders.errors.FactGraphSaveException;
import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.stateapi.domain.StateApiCreateAuthorizationCodeRequest;
import gov.irs.directfile.api.stateapi.domain.StateApiCreateAuthorizationTokenRequest;
import gov.irs.directfile.api.stateapi.domain.StateProfile;
import gov.irs.directfile.api.stateapi.domain.export.*;
import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.models.StateOrProvince;

@Service
public class StateApiService {

    private final WebClient webClient;

    private final FactGraphService factGraphService;
    private final StateApiEndpointProperties stateApiEndpointProperties;

    private static final String FILING_STATUS = "/filingStatus";

    StateApiService(FactGraphService factGraphService, StateApiEndpointProperties stateApiEndpointProperties) {
        webClient = WebClient.builder()
                .baseUrl(stateApiEndpointProperties.getBaseUrl())
                .build();
        this.factGraphService = factGraphService;
        this.stateApiEndpointProperties = stateApiEndpointProperties;
    }

    public AuthCodeResponse getAuthorizationCode(StateApiCreateAuthorizationCodeRequest requestBody) {
        return webClient
                .post()
                .uri("/authorization-code")
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(AuthCodeResponse.class)
                .block();
    }

    public String getAuthorizationToken(StateApiCreateAuthorizationTokenRequest requestBody) {
        return webClient
                .post()
                .uri(stateApiEndpointProperties.getV2AuthTokenPath())
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(String.class)
                .block();
    }

    public StateProfile getStateProfile(StateOrProvince stateCode) {
        return webClient
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path("/state-profile")
                        .queryParam("stateCode", stateCode)
                        .build())
                .retrieve()
                .bodyToMono(StateProfile.class)
                .block();
    }

    public ExportableFacts getExportToStateFacts(final Graph graph)
            throws JsonProcessingException, FactGraphSaveException {
        return new ExportableFacts();
    }
}
