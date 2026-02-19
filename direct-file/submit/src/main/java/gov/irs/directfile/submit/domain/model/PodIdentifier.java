package gov.irs.directfile.submit.domain.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@Getter
@Setter
@Entity(name = "PodIdentifier")
@Table(name = "pod_identifier")
public class PodIdentifier {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "asid", length = 50)
    private String asid;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "region", length = 50)
    private String region;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "podId", length = 255)
    private String podId;
}
