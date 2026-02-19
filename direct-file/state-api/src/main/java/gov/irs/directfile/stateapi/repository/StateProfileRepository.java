package gov.irs.directfile.stateapi.repository;

import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.model.StateProfile;

@Repository
public interface StateProfileRepository extends R2dbcRepository<StateProfile, Integer> {
    Mono<StateProfile> getByAccountId(String accountId);

    Mono<StateProfile> getByStateCode(String stateCode);
}
