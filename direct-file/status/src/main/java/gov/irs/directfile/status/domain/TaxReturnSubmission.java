package gov.irs.directfile.status.domain;

import java.util.Date;
import java.util.UUID;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP", "EI_EXPOSE_REP2"},
        justification = "Initial SpotBugs Setup")
@Entity(name = "TaxReturnSubmission")
@Table(name = "tax_return_submission")
@Getter
@Setter
public class TaxReturnSubmission {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private UUID taxReturnId;

    @Column(length = 20)
    private String submissionId;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;

    public TaxReturnSubmission(UUID taxReturnId, String submissionId) {
        this.taxReturnId = taxReturnId;
        this.submissionId = submissionId;
    }

    public TaxReturnSubmission() {}
}
