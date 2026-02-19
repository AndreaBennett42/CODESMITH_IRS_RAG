package gov.irs.directfile.status.domain;

import java.util.Date;
import java.util.List;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP", "EI_EXPOSE_REP2"},
        justification = "Initial SpotBugs Setup")
@Entity(name = "Completed")
@Getter
@Setter
public class Completed {
    @Id
    @Column(length = 20)
    private String submissionId;

    @Column(length = 20)
    private String status;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;

    @Getter
    @ManyToMany(fetch = FetchType.EAGER)
    @JoinTable(
            name = "completed-errors",
            joinColumns = {
                @JoinColumn(
                        name = "submissionId",
                        referencedColumnName = "submissionId",
                        foreignKey = @ForeignKey(name = "completed_error_submission_id_fk"))
            },
            inverseJoinColumns = {
                @JoinColumn(
                        name = "mefErrorCode",
                        referencedColumnName = "meferror_code",
                        foreignKey = @ForeignKey(name = "completed_error_mef_error_code_fk"))
            })
    private List<Error> errors;
}
