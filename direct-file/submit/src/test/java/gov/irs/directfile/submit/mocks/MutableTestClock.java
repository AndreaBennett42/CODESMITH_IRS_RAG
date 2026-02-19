package gov.irs.directfile.submit.mocks;

import java.time.Clock;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.temporal.TemporalAmount;

// Taken from: https://jonasg.io/posts/how-to-effectively-test-time-dependent-code/
// This class should allow us to easily test moving forward in time for time based operations.
public class MutableTestClock extends Clock {

    private Instant instant;

    private final ZoneId zone;

    public MutableTestClock(Instant instant, ZoneId zone) {
        this.instant = instant;
        this.zone = zone;
    }

    /**
     * Default constructor that creates a clock at the current instant in the UTC timezone.
     *
     * */
    public MutableTestClock() {
        this.instant = Instant.now();
        this.zone = ZoneId.of("UTC");
    }

    @Override
    public ZoneId getZone() {
        return zone;
    }

    @Override
    public Clock withZone(ZoneId zone) {
        return new MutableTestClock(instant, zone);
    }

    @Override
    public Instant instant() {
        return instant;
    }

    public void fastForward(TemporalAmount temporalAmount) {
        set(instant().plus(temporalAmount));
    }

    public void rewind(TemporalAmount temporalAmount) {
        set(instant().minus(temporalAmount));
    }

    public void set(Instant instant) {
        this.instant = instant;
    }

    public static MutableTestClock fixed(Instant instant, ZoneId zone) {
        return new MutableTestClock(instant, zone);
    }

    public static MutableTestClock fixed(OffsetDateTime offsetDateTime) {
        return fixed(offsetDateTime.toInstant(), offsetDateTime.getOffset());
    }
}
