package gov.irs.directfile.api.io.storagelocations;

import java.util.UUID;

public class StorageLocationBuilder {
    private static String fileExtension = ".pdf";

    public static String getTaxReturnLocation(int taxFilingYear, UUID taxReturnId) {
        return taxFilingYear + "/taxreturns/" + taxReturnId;
    }

    public static String getTaxReturnDocumentFilename(String baseName, int year, String languageCode) {
        return baseName + year + languageCode + fileExtension;
    }

    public static String getTaxReturnDocumentLocation(
            int taxFilingYear, UUID taxReturnId, String baseName, String languageCode) {
        return getTaxReturnLocation(taxFilingYear, taxReturnId) + "/userdocuments/"
                + getTaxReturnDocumentFilename(baseName, taxFilingYear, languageCode);
    }

    public static String getSubmissionLocation(int taxFilingYear, UUID taxReturnId) {
        return getTaxReturnLocation(taxFilingYear, taxReturnId) + "/submissions/";
    }
}
