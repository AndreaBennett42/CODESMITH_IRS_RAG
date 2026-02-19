package gov.irs.directfile.submit.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.submit.domain.model.PodIdentifier;

public interface PodIdentifierRepository extends CrudRepository<PodIdentifier, Long> {

    @Query(value = "SELECT asid from pod_identifier WHERE pod_id = :pod_id", nativeQuery = true)
    Optional<String> findAsidByPodId(String pod_id);
}
