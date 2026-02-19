package gov.irs.directfile.stateapi.repository;

import java.util.UUID;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.configuration.DirectFileEndpointProperties;
import gov.irs.directfile.stateapi.model.TaxReturnStatus;

@Component
@ConditionalOnProperty(name = "direct-file.status.mock", havingValue = "false", matchIfMissing = true)
@Slf4j
@SuppressWarnings("PMD.AvoidDuplicateLiterals")
public class DirectFileClientImpl implements DirectFileClient {

    private DirectFileBackendService directFileBackendService;

    public DirectFileClientImpl(DirectFileEndpointProperties dfEndpointProperties) {
        directFileBackendService = new DirectFileBackendService(dfEndpointProperties);
    }

    @Override
    public Mono<TaxReturnStatus> getStatus(int taxYear, UUID taxReturnId, String submissionId) {
        return directFileBackendService
                .getStatus(taxYear, taxReturnId, submissionId)
                .map(ts -> {
                    if (!ts.exists()) {
                        log.warn(
                                "getStatus: E_TAX_RETURN_NOT_FOUND (possibly due to replication delay across regions, expecting retry from user). taxReturnId={}, submissionId={}, tax year={}",
                                taxReturnId,
                                submissionId,
                                taxYear);
                    }
                    return new TaxReturnStatus(ts.status().toLowerCase(), ts.exists());
                })
                .doOnError(e -> {
                    log.error(
                            "getStatus() failed for taxYear={}, taxReturnId={}, submissionId={}. Exception: {}. Error: {}",
                            taxYear,
                            taxReturnId.toString(),
                            submissionId,
                            e.getClass().getName(),
                            e.getMessage());
                });
    }
}
