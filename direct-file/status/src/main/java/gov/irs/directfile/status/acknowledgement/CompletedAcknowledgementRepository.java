package gov.irs.directfile.status.acknowledgement;

import java.util.Optional;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.status.domain.Completed;

public interface CompletedAcknowledgementRepository extends CrudRepository<Completed, String> {

    @Query(value = "SELECT * FROM completed WHERE submission_id = :submissionId LIMIT 1", nativeQuery = true)
    Optional<Completed> GetCompletedSubmission(String submissionId);
}
