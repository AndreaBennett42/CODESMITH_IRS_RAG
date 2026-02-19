package gov.irs.directfile.status.domain;

import java.util.Date;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.*;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

@SuppressFBWarnings(
        value = {"NM_FIELD_NAMING_CONVENTION", "EI_EXPOSE_REP", "EI_EXPOSE_REP2"},
        justification = "Initial SpotBugs Setup")
@Entity
@Getter
@Setter
public class ToolkitError {
    @Id
    private String submissionId;

    @NotNull private String errorMessage;

    @NotNull private String errorName;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;
}
