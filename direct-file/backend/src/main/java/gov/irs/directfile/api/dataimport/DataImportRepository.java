package gov.irs.directfile.api.dataimport;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import gov.irs.directfile.api.dataimport.model.PopulatedData;

public interface DataImportRepository extends JpaRepository<PopulatedData, UUID> {
    @Query(
            value =
                    "WITH descending as (SELECT * FROM populated_data WHERE taxreturn_id = :taxReturnId ORDER BY created_at DESC) SELECT DISTINCT ON (source) * from descending",
            nativeQuery = true)
    List<PopulatedData> findLatestSourcesByTaxReturnId(UUID taxReturnId);

    Optional<PopulatedData> findByTaxReturnIdAndSource(UUID taxReturnId, String source);
}
