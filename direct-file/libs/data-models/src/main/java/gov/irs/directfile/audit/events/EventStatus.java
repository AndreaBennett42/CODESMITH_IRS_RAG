package gov.irs.directfile.audit.events;

public enum EventStatus {
    SUCCESS("00"),
    FAILURE("01");

    private final String value;

    EventStatus(String value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return value;
    }
}
