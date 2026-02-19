package gov.irs.directfile.api.taxreturn;

import java.util.Date;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.domain.Limit;
import org.springframework.data.domain.ScrollPosition;
import org.springframework.data.domain.Window;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.api.taxreturn.models.SubmissionEvent;

public interface SubmissionEventRepository extends CrudRepository<SubmissionEvent, UUID> {
    @Query(
            value = "SELECT * FROM submission_events WHERE id = :submissionId ORDER BY created_at DESC",
            nativeQuery = true)
    List<SubmissionEvent> findEventsBySubmissionId(UUID submissionId);

    @Query(
            value =
                    "SELECT * FROM submission_events WHERE taxreturn_submission_id = :submissionId ORDER BY created_at DESC LIMIT 1",
            nativeQuery = true)
    Optional<SubmissionEvent> findLatestEvent(UUID submissionId);

    @Query(
            value =
                    "SELECT count(*) FROM submission_events WHERE taxreturn_submission_id = :submissionId AND event_type='failed'",
            nativeQuery = true)
    int countFailedEvents(UUID submissionId);

    @Query(
            value =
                    """
                SELECT se.*
                FROM submission_events se
                JOIN taxreturn_submissions trs ON se.taxreturn_submission_id = trs.id
                WHERE trs.taxreturn_id = :taxReturnId
                ORDER BY se.created_at DESC LIMIT 1
                """,
            nativeQuery = true)
    Optional<SubmissionEvent> getLatestSubmissionEventByTaxReturnId(UUID taxReturnId);

    @Query(
            value =
                    """
                SELECT se.*
                FROM submission_events se
                JOIN taxreturn_submissions trs ON se.taxreturn_submission_id = trs.id
                WHERE trs.taxreturn_id = :taxReturnId
                      AND se.event_type = 'accepted'
                ORDER BY se.created_at DESC LIMIT 1
                """,
            nativeQuery = true)
    Optional<SubmissionEvent> getLatestAcceptedSubmissionEventForTaxReturnId(UUID taxReturnId);

    // Based on Spring Query Method Docs:
    // https://docs.spring.io/spring-data/jpa/reference/repositories/query-methods-details.html
    @EntityGraph(attributePaths = {"submission"})
    Window<SubmissionEvent> findByEventTypeInAndCreatedAtBetween(
            List<String> eventTypes, Date startDate, Date endDate, Limit limit, ScrollPosition scrollPosition);

    @EntityGraph(attributePaths = {"submission"})
    List<SubmissionEvent> findByEventTypeInAndSubmissionSubmissionIdIn(
            List<String> eventTypes, List<String> submissionIds);
}
