package gov.irs.directfile.stateapi.events;

import java.time.Instant;

import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class Event {
    @NotNull private final Instant timestamp = Instant.now();

    @NotNull private final EventStatus eventStatus;

    @NotNull private final EventId eventId;

    @NotNull private final String responseStatusCode;

    @NotNull private String taxPeriod;

    @NotNull private String userType;

    @NotNull private String remoteAddress;

    @NotNull private String stateId;

    @NotNull private String taxReturnId;

    private final String eventErrorMessage;
    private EventDetail detail;

    @NotNull @Builder.Default
    private String eventType = "STATE_API";

    @NotNull @Builder.Default
    private final boolean cyberOnly = true;
}
