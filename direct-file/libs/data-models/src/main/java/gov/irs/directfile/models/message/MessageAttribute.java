package gov.irs.directfile.models.message;

public enum MessageAttribute {
    VERSION("VERSION"),
    USER_ID("USER-ID"),
    TAX_RETURN_ID("TAX-RETURN-ID"),
    EXTERNAL_ID("EXTERNAL-ID"),
    TIN_CIPHER_TEXT("TIN-CIPHER-TEXT"),
    POPULATED_DATA_SOURCE("POPULATED-DATA-SOURCE"),
    POPULATED_DATA_TAGS("POPULATED-DATA-TAGS"),
    RAW_DATA("RAW-DATA"),
    TAX_YEAR("TAX-YEAR");

    private final String name;

    MessageAttribute(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
