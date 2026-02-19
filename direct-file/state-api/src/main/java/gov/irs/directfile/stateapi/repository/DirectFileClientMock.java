package gov.irs.directfile.stateapi.repository;

import java.util.UUID;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.model.TaxReturnStatus;

@Component
@Slf4j
@ConditionalOnProperty(name = "direct-file.status.mock", havingValue = "true", matchIfMissing = false)
public class DirectFileClientMock implements DirectFileClient {

    @Override
    public Mono<TaxReturnStatus> getStatus(int taxYear, UUID taxReturnId, String submissionId) {
        log.info("Enter Mock getStatus submissionId={}, taxReturnId={}", submissionId, taxReturnId);

        String idString = taxReturnId.toString();
        if (idString.startsWith("a")) return Mono.just(new TaxReturnStatus(STATUS_ACCEPTED, true));
        else if (idString.startsWith("b")) return Mono.just(new TaxReturnStatus(STATUS_REJECTED, true));
        else return Mono.just(new TaxReturnStatus(STATUS_PENDING, true));
    }
}
