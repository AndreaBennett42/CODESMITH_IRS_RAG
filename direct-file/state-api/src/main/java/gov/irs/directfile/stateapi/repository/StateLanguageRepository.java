package gov.irs.directfile.stateapi.repository;

import org.springframework.data.r2dbc.repository.R2dbcRepository;
import reactor.core.publisher.Flux;

import gov.irs.directfile.stateapi.model.StateLanguage;

public interface StateLanguageRepository extends R2dbcRepository<StateLanguage, Long> {
    Flux<StateLanguage> getAllByStateProfileId(Long stateProfileId);
}
