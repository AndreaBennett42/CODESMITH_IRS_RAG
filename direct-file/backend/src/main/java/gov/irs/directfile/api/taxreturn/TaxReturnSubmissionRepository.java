package gov.irs.directfile.api.taxreturn;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;

public interface TaxReturnSubmissionRepository extends CrudRepository<TaxReturnSubmission, UUID> {
    @Query(
            value = "SELECT * FROM taxreturn_submissions WHERE taxreturn_id = :taxReturnId ORDER BY created_at DESC",
            nativeQuery = true)
    List<TaxReturnSubmission> findSubmissionsByTaxReturnId(UUID taxReturnId);

    @Query(
            value =
                    "SELECT * FROM taxreturn_submissions WHERE taxreturn_id = :taxReturnId ORDER BY created_at DESC LIMIT 1",
            nativeQuery = true)
    Optional<TaxReturnSubmission> findLatestTaxReturnSubmissionByTaxReturnId(UUID taxReturnId);

    @Query(value = "SELECT * FROM taxreturn_submissions WHERE submission_id in :submissionIds", nativeQuery = true)
    List<TaxReturnSubmission> findAllBySubmissionIds(List<String> submissionIds);

    @Query(value = "SELECT * FROM taxreturn_submissions WHERE submission_id = :submissionId", nativeQuery = true)
    Optional<TaxReturnSubmission> findSubmissionBySubmissionId(String submissionId);

    @Query(
            value = "SELECT * FROM taxreturn_submissions WHERE taxreturn_id = :taxReturnId ORDER BY CREATED_AT DESC",
            nativeQuery = true)
    List<TaxReturnSubmission> findAllTaxReturnSubmissionsByTaxReturnId(UUID taxReturnId);

    @Query(
            value =
                    "WITH ordered_submissions AS ( SELECT * FROM taxreturn_submissions WHERE taxreturn_id IN :taxReturnIds ORDER BY created_at DESC ) "
                            + "SELECT DISTINCT ON (taxreturn_id) "
                            + "* FROM ordered_submissions; ",
            nativeQuery = true)
    List<TaxReturnSubmission> findLatestTaxReturnSubmissions(List<UUID> taxReturnIds);

    @Query(
            value =
                    "WITH ordered_submissions AS ( SELECT * FROM taxreturn_submissions WHERE taxreturn_id IN :taxReturnIds AND "
                            + "id NOT IN (SELECT taxreturn_submission_id from submission_events where LOWER(event_type) = 'submitted')"
                            + " ORDER BY created_at DESC ) "
                            + "SELECT DISTINCT ON (taxreturn_id) "
                            + "* FROM ordered_submissions; ",
            nativeQuery = true)
    List<TaxReturnSubmission> findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(List<UUID> taxReturnIds);

    @Query(value = "SELECT id FROM taxreturn_submissions WHERE submission_id IN :submissionIds", nativeQuery = true)
    List<UUID> findIdBySubmissionId(List<String> submissionIds);

    @Query(
            value =
                    "WITH non_terminal_events AS ( SELECT taxreturn_submission_id FROM submission_events WHERE taxreturn_submission_id IN :taxReturnSubmissionIds "
                            + "GROUP BY taxreturn_submission_id HAVING sum( CASE WHEN event_type IN ('accepted', 'rejected') THEN 1 ELSE 0 END ) = 0 ) "
                            + "SELECT taxreturn_submissions.* FROM taxreturn_submissions "
                            + "INNER JOIN non_terminal_events ON taxreturn_submissions.id = non_terminal_events.taxreturn_submission_id "
                            + "WHERE taxreturn_submissions.id IN :taxReturnSubmissionIds",
            nativeQuery = true)
    List<TaxReturnSubmission> findAllWithoutTerminalEventsByTaxReturnSubmissionId(List<UUID> taxReturnSubmissionIds);

    @Query(
            value =
                    "SELECT taxreturn_submissions.* FROM taxreturn_submissions INNER JOIN submission_events ON taxreturn_submissions.id = submission_events.taxreturn_submission_id "
                            + "WHERE taxreturn_submissions.id IN :taxReturnSubmissionIds AND event_type = :eventType",
            nativeQuery = true)
    List<TaxReturnSubmission> findAllWithTerminalEventsByTaxReturnSubmissionIdByEventType(
            List<UUID> taxReturnSubmissionIds, String eventType);

    @Query("SELECT CASE WHEN 1 <= (SELECT count(e.id) "
            + "                     FROM SubmissionEvent e "
            + "                     WHERE (e.eventType != 'processing' AND e.eventType != 'submitted' AND e.eventType!='accepted' AND e.eventType!='reminderstatetax') "
            + "                       AND e.submission.id = s.id) THEN TRUE "
            + "           WHEN 0 = (SELECT count(e.id) "
            + "                     FROM SubmissionEvent e "
            + "                     WHERE e.submission.id = s.id) THEN TRUE "
            + "           ELSE FALSE "
            + "      END "
            + "FROM TaxReturnSubmission s "
            + "WHERE s.taxReturn.id = :taxReturnId "
            + "ORDER BY s.createdAt DESC "
            + "LIMIT 1")
    Optional<Boolean> isTaxReturnEditable(UUID taxReturnId);
}
