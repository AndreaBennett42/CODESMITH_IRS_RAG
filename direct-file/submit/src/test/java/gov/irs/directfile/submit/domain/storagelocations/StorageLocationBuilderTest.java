package gov.irs.directfile.submit.domain.storagelocations;

import java.util.UUID;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class StorageLocationBuilderTest {
    @Test
    public void givenTaxFilingYearAndTaxReturnIdAndSubmissionId_whenGetSubmissionLocation_thenShouldSucceed() {
        // given
        int taxFilingYear = 2020;
        UUID taxReturnId = UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161");
        String submissionId = "78126520231005000008";

        // when
        String actual = StorageLocationBuilder.getSubmissionLocation(taxFilingYear, taxReturnId, submissionId, "xml");

        // then
        assertEquals(
                "2020/taxreturns/738fc2dc-88f9-4b5c-ace9-c602509ba161/submissions/78126520231005000008.xml", actual);
    }

    @Test
    public void givenTaxFilingYearAndTaxReturnId_whenGetSubmissionLocation_thenShouldSucceed() {
        // given
        int taxFilingYear = 2020;
        UUID taxReturnId = UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161");

        // when
        String actual = StorageLocationBuilder.getSubmissionLocation(taxFilingYear, taxReturnId);

        // then
        assertEquals("2020/taxreturns/738fc2dc-88f9-4b5c-ace9-c602509ba161/submissions/", actual);
    }
}
