package gov.irs.directfile.status.acknowledgement;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.status.domain.Pending;

public interface PendingAcknowledgementRepository extends CrudRepository<Pending, String> {

    @Query(value = "SELECT * FROM pending WHERE submission_id = :submissionId LIMIT 1", nativeQuery = true)
    Optional<Pending> GetPendingSubmission(String submissionId);

    @Query(value = "SELECT * FROM pending WHERE pod_id = :podId ORDER BY created_at asc", nativeQuery = true)
    List<Pending> findAllByPodId(String podId);
}
