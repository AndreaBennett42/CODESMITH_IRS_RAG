package gov.irs.directfile.stateapi.repository.facts;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.configuration.DirectFileEndpointProperties;
import gov.irs.directfile.stateapi.model.GetStateExportedFactsResponse;
import gov.irs.directfile.stateapi.repository.DirectFileBackendService;

@Component
@ConditionalOnProperty(name = "direct-file.exported-facts.mock", havingValue = "false", matchIfMissing = true)
@Slf4j
public class ExportedFactsClientImpl implements ExportedFactsClient {

    private DirectFileBackendService directFileBackendService;

    public ExportedFactsClientImpl(DirectFileEndpointProperties dfEndpointProperties) {
        directFileBackendService = new DirectFileBackendService(dfEndpointProperties);
    }

    @Override
    public Mono<GetStateExportedFactsResponse> getExportedFacts(
            String submissionId, String stateCode, String accountId) {
        return directFileBackendService
                .getExportedFacts(submissionId, stateCode, accountId)
                .doOnSuccess(ef -> log.info(
                        "getExportedFacts runs successfully for accountId={}, submissionId={}",
                        accountId,
                        submissionId))
                .doOnError(e -> {
                    log.error(
                            "getExportedFacts() failed for accountId={}, submissionId={}, {}, error: {}",
                            accountId,
                            submissionId,
                            e.getClass().getName(),
                            e.getMessage());
                });
    }
}
