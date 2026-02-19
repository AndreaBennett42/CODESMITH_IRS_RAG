package gov.irs.directfile.models.message;

public enum MessageHeaderAttribute {
    VERSION("version"),
    PRODUCER("producer"),
    CONSUMER("consumer"),

    METADATA("metadata");

    private final String attribute;

    MessageHeaderAttribute(String attribute) {
        this.attribute = attribute;
    }

    public String getAttribute() {
        return attribute;
    }
}
