package gov.irs.directfile.stateapi.repository;

import org.springframework.data.r2dbc.repository.R2dbcRepository;
import reactor.core.publisher.Flux;

import gov.irs.directfile.stateapi.model.StateRedirect;

public interface StateRedirectRepository extends R2dbcRepository<StateRedirect, Long> {
    Flux<StateRedirect> getAllByStateProfileId(Long stateProfileId);
}
