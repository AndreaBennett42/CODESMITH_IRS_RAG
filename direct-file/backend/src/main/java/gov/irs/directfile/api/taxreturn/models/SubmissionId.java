package gov.irs.directfile.api.taxreturn.models;

import java.util.*;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@Getter
@Entity
@Table(name = "submission_ids")
@SuppressFBWarnings(value = {"EI_EXPOSE_REP"})
public class SubmissionId {
    @Id
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Setter
    @Column(
            columnDefinition = "varchar",
            name = "submission_id",
            nullable = false,
            updatable = false,
            length = 20,
            unique = true)
    private String submissionId;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;
}
