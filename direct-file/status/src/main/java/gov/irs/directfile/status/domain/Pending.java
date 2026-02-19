package gov.irs.directfile.status.domain;

import java.util.Date;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP", "EI_EXPOSE_REP2", "NM_FIELD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@Entity(name = "Pending")
@Getter
@Setter
public class Pending {
    @Id
    @Column(length = 20)
    private String submissionId;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "pod_id", length = 255)
    private String podId;

    public Pending() {}

    public Pending(final String submissionId) {
        this.submissionId = submissionId;
    }

    public Pending(String submissionId, String podId) {
        this.submissionId = submissionId;
        this.podId = podId;
    }
}
