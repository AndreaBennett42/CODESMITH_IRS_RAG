package gov.irs.directfile.audit.events;

import lombok.Builder;
import lombok.Getter;
import org.springframework.lang.NonNull;

@Getter
@Builder
public class Event {
    @NonNull private final EventStatus eventStatus;

    @NonNull private final EventId eventId;

    @NonNull private final EventPrincipal eventPrincipal;

    private final String eventErrorMessage;

    private String detail;
    private String email;
    private String mefSubmissionId;
    private String userTin;
    private TinType userTinType;

    @Builder.Default
    private final boolean cyberOnly = true;
}
