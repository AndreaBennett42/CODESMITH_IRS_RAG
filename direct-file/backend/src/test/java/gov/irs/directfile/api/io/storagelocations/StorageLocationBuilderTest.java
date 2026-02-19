package gov.irs.directfile.api.io.storagelocations;

import java.util.UUID;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class StorageLocationBuilderTest {
    @Test
    public void givenTaxFilingYearAndTaxReturnId_whenGetTaxReturnLocation_thenShouldSucceed() {
        // given
        int taxFilingYear = 2020;
        UUID taxReturnId = UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161");

        // when
        String actual = StorageLocationBuilder.getTaxReturnLocation(taxFilingYear, taxReturnId);

        // then
        assertEquals("2020/taxreturns/738fc2dc-88f9-4b5c-ace9-c602509ba161", actual);
    }

    @Test
    public void givenTaxFilingYearAndTaxReturnIdAndLanguageCode_whenGetTaxReturnDocumentLocation_thenShouldSucceed() {
        // given
        int taxFilingYear = 2020;
        UUID taxReturnId = UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161");
        String languageCode = "en";
        String formName = "taxreturn";

        // when
        String actual =
                StorageLocationBuilder.getTaxReturnDocumentLocation(taxFilingYear, taxReturnId, formName, languageCode);

        // then
        assertEquals("2020/taxreturns/738fc2dc-88f9-4b5c-ace9-c602509ba161/userdocuments/taxreturn2020en.pdf", actual);
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
