package gov.irs.directfile.status.acknowledgement;

import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import gov.irs.directfile.status.domain.TaxReturnSubmission;

import static org.junit.jupiter.api.Assertions.*;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@DataJpaTest
class TaxReturnSubmissionRepositoryTest {
    @Autowired
    TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Test
    public void canSaveAndRetrieveTaxReturnSubmissions() {
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "12345678901234567890";
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, submissionId));

        Optional<String> result = taxReturnSubmissionRepository.getLatestSubmissionIdByTaxReturnId(taxReturnId);

        assertTrue(result.isPresent());
        assertEquals(result.get(), submissionId);

        // Note: Hibernate disables batching for entities having an identity PK, so no batching to test for here
    }

    @Test
    public void returnsEmptyOptionalWhenTryingToFindSubmissionIdOfTaxReturnThatDoesNotExist() {
        UUID taxReturnId = UUID.randomUUID();
        Optional<String> result = taxReturnSubmissionRepository.getLatestSubmissionIdByTaxReturnId(taxReturnId);

        assertTrue(result.isEmpty());
    }
}
