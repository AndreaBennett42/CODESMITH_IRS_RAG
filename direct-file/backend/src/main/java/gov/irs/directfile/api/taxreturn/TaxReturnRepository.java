package gov.irs.directfile.api.taxreturn;

import java.util.Date;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.domain.KeysetScrollPosition;
import org.springframework.data.domain.Limit;
import org.springframework.data.domain.ScrollPosition;
import org.springframework.data.domain.Window;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.api.taxreturn.models.TaxReturn;

public interface TaxReturnRepository extends CrudRepository<TaxReturn, UUID> {
    @Query("SELECT t FROM TaxReturn t JOIN t.owners o WHERE o.id = :userId ORDER BY t.taxYear DESC")
    List<TaxReturn> findByUserId(UUID userId);

    @Query("SELECT t FROM TaxReturn t JOIN t.owners o WHERE o.id = :userId AND t.id = :id")
    Optional<TaxReturn> findByIdAndUserId(UUID id, UUID userId);

    @Query("SELECT t FROM TaxReturn t JOIN t.owners o WHERE o.id = :userId AND t.taxYear = :taxYear")
    Optional<TaxReturn> findByUserIdAndTaxYear(UUID userId, int taxYear);

    @Query(value = "SELECT t FROM TaxReturn t WHERE t.id in :taxReturnIds")
    List<TaxReturn> findAllByTaxReturnIds(List<UUID> taxReturnIds);

    // Based on Spring Query Method Docs:
    // https://docs.spring.io/spring-data/jpa/reference/repositories/query-methods-details.html
    Window<TaxReturn> findByTaxYearAndCreatedAtBetweenOrderByCreatedAtAsc(
            Limit limit, ScrollPosition scrollPosition, int taxYear, Date createdStart, Date createdEnd);

    // Scrolling Reference Docs:
    // https://docs.spring.io/spring-data/jpa/reference/data-commons/repositories/scrolling.html#repositories.scrolling.keyset
    // Query Method Docs:
    // https://docs.spring.io/spring-data/jpa/docs/current-SNAPSHOT/reference/html/#jpa.query-methods.query-creation
    Window<SimpleTaxReturnProjection> findByTaxYearAndSubmitTimeIsNullAndCreatedAtBetweenOrderByCreatedAtAsc(
            Limit limit, int taxYear, Date createdStart, Date createdEnd, KeysetScrollPosition position);
}
