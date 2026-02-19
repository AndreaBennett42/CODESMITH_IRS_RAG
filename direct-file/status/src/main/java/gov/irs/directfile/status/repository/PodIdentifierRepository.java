package gov.irs.directfile.status.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.status.domain.PodIdentifier;

public interface PodIdentifierRepository extends CrudRepository<PodIdentifier, Long> {

    @Query(value = "SELECT asid from pod_identifier WHERE pod_id = :pod_id", nativeQuery = true)
    Optional<String> findAsidByPodId(String pod_id);
}
