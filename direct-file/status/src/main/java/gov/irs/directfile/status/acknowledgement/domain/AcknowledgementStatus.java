package gov.irs.directfile.status.acknowledgement.domain;

import java.util.Date;
import java.util.List;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import gov.irs.directfile.models.RejectedStatus;

@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP", "EI_EXPOSE_REP2"},
        justification = "Initial SpotBugs Setup")
@Getter
@Setter
@AllArgsConstructor
public class AcknowledgementStatus {
    private Status status;
    private String translationKey;
    private List<RejectedStatus> rejectionCodes;
    private Date createdAt;
}
