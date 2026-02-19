package gov.irs.directfile.api.events;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.Builder;
import lombok.Getter;
import org.springframework.lang.NonNull;

@Getter
@Builder
@SuppressFBWarnings(value = "CT_CONSTRUCTOR_THROW", justification = "Java 21 update")
public class Event {
    @NonNull private final EventStatus eventStatus;

    @NonNull private final EventId eventId;

    @NonNull private final EventPrincipal eventPrincipal;

    private final String eventErrorMessage;

    private String email;
    private String mefSubmissionId;
    private String taxPeriod;
    private String userTin;
    private String userTinType;

    @Builder.Default
    private final boolean cyberOnly = true;
}
