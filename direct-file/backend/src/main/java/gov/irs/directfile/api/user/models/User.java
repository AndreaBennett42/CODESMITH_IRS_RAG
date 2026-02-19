package gov.irs.directfile.api.user.models;

import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.type.SqlTypes;

import gov.irs.directfile.api.taxreturn.models.TaxReturn;

@Getter
@Entity
@Table(
        name = "users",
        indexes = {@Index(name = "users_external_id_unique_idx", columnList = "externalId", unique = true)})
@SuppressFBWarnings(value = {"EI_EXPOSE_REP"})
public class User {
    @Id
    @GeneratedValue(generator = "UUID4")
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Setter
    @Column(nullable = false)
    private boolean accessGranted = false;

    @Column(nullable = false, updatable = false)
    private UUID externalId;

    @Setter
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(name = "email", columnDefinition = "varchar")
    private String emailCipherText;

    @Setter
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(name = "tin", columnDefinition = "varchar")
    private String tinCipherText;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;

    @Getter
    @Column(name = "updated_at")
    @UpdateTimestamp
    private Date updatedAt;

    @ManyToMany
    @JoinTable(
            name = "taxreturn_owners",
            joinColumns = {
                @JoinColumn(
                        name = "owner_id",
                        referencedColumnName = "id",
                        nullable = false,
                        foreignKey = @ForeignKey(name = "owner_id_fkey"))
            },
            inverseJoinColumns = {
                @JoinColumn(
                        name = "taxreturn_id",
                        referencedColumnName = "id",
                        nullable = false,
                        foreignKey =
                                @ForeignKey(
                                        name = "taxreturn_id_fkey",
                                        foreignKeyDefinition =
                                                "FOREIGN KEY (taxreturn_id) REFERENCES taxreturns ON DELETE CASCADE"))
            },
            indexes = @Index(columnList = "owner_id, taxreturn_id", unique = true))
    private Set<TaxReturn> taxReturns = new HashSet<>();

    protected User() {}

    public User(final UUID externalId) {
        this.externalId = externalId;
    }

    public void addTaxReturn(TaxReturn newTaxReturn, boolean sync) {
        this.taxReturns.add(newTaxReturn);
        if (sync) newTaxReturn.addOwner(this, false);
    }

    public void addTaxReturn(TaxReturn newTaxReturn) {
        this.addTaxReturn(newTaxReturn, true);
    }
}
