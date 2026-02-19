package gov.irs.directfile.stateapi.repository;

import org.springframework.data.r2dbc.repository.R2dbcRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.model.AuthorizationCode;

@Repository
public interface AuthorizationCodeRepository extends R2dbcRepository<AuthorizationCode, Integer> {
    Mono<AuthorizationCode> getByAuthorizationCode(@Param("authorizationCode") String authDigest);
}
