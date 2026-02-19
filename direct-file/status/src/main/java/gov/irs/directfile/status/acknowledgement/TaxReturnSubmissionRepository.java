package gov.irs.directfile.status.acknowledgement;

import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.status.domain.TaxReturnSubmission;

public interface TaxReturnSubmissionRepository extends CrudRepository<TaxReturnSubmission, Long> {
    @Query(
            value =
                    "SELECT submission_id FROM tax_return_submission WHERE tax_return_id = :taxReturnId ORDER BY CREATED_AT DESC LIMIT 1",
            nativeQuery = true)
    Optional<String> getLatestSubmissionIdByTaxReturnId(UUID taxReturnId);

    @Query(
            value =
                    """
                SELECT trs.submission_id
                FROM tax_return_submission trs
                JOIN completed c ON trs.submission_id = c.submission_id
                WHERE trs.tax_return_id = :taxReturnId
                      AND LOWER(c.status) = 'accepted'
                ORDER BY c.created_at DESC LIMIT 1
                """,
            nativeQuery = true)
    Optional<String> getLatestAcceptedSubmissionIdForTaxReturnId(UUID taxReturnId);

    @Query(
            value =
                    """
                SELECT trs.submission_id
                FROM tax_return_submission trs
                JOIN completed c ON trs.submission_id = c.submission_id
                WHERE
                    trs.tax_return_id = (
                        SELECT tax_return_id
                        FROM tax_return_submission
                        WHERE submission_id = :submissionId
                    )
                    AND LOWER(c.status) = 'accepted'
                ORDER BY c.created_at DESC LIMIT 1
                """,
            nativeQuery = true)
    Optional<String> getLatestAcceptedSubmissionIdOfParentTaxReturn(String submissionId);
}
