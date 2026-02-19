package gov.irs.directfile.api.taxreturn.models;

import java.util.*;

import com.fasterxml.jackson.annotation.JsonBackReference;
import jakarta.persistence.*;
import jakarta.persistence.CascadeType;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.*;
import org.hibernate.type.SqlTypes;

import gov.irs.directfile.api.dataimport.gating.DataImportBehavior;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.models.FactTypeWithItem;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Getter
@Entity
@Table(name = "taxreturns")
@EntityListeners({TaxReturnEntityListener.class})
public class TaxReturn implements TaxReturnEntity {
    private static final String DD = "DD-to-force-reencryption-by-entity-listener";

    @Id
    @GeneratedValue(generator = "UUID4")
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Getter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;

    @Getter
    @Column(name = "updated_at")
    @UpdateTimestamp
    private Date updatedAt;

    @Setter
    @Column(nullable = false)
    private int taxYear;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(name = "facts", columnDefinition = "varchar")
    @Setter
    private String factsCipherText;

    @Transient
    private Map<String, FactTypeWithItem> facts;

    public void setFacts(Map<String, FactTypeWithItem> facts) {
        this.setFactsCipherText(DD);
        this.facts = facts;
    }

    @Override
    public void setFactsWithoutDirtyingEntity(Map<String, FactTypeWithItem> facts) {
        this.facts = facts;
    }

    @ManyToMany(mappedBy = "taxReturns")
    @JsonBackReference
    private Set<User> owners = new HashSet<>();

    @Column
    private Date submitTime;

    @Column
    private UUID submitUserId;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Setter
    @Column(name = "store", columnDefinition = "varchar")
    private String storeCipherText;

    @Setter
    @Column
    private Boolean surveyOptIn;

    @Setter
    @Column
    private String dataImportBehavior;

    @Transient
    private String store;

    public void setStore(String store) {
        this.setStoreCipherText(DD);
        this.store = store;
    }

    @Override
    public void setStoreWithoutDirtyingEntity(String store) {
        this.store = store;
    }

    @OneToMany(mappedBy = "taxReturn", cascade = CascadeType.ALL, targetEntity = TaxReturnSubmission.class)
    private Set<TaxReturnSubmission> taxReturnSubmissions = new HashSet<>();

    public Date getMostRecentSubmitTime() {
        return submitTime;
    }

    public void setMostRecentSubmitTime(Date submitTime) {
        this.submitTime = submitTime;
    }

    public UUID getMostRecentSubmitUserId() {
        return submitUserId;
    }

    public void setMostRecentSubmitUserId(UUID submitUserId) {
        this.submitUserId = submitUserId;
    }

    public TaxReturnSubmission addTaxReturnSubmission() {
        TaxReturnSubmission taxReturnSubmission = new TaxReturnSubmission();
        if (getMostRecentSubmitTime() == null) {
            setMostRecentSubmitTime(new Date());
        }
        taxReturnSubmission.setCreatedAt(getMostRecentSubmitTime());
        taxReturnSubmission.setFacts(facts);
        taxReturnSubmission.setSubmitUserId(getMostRecentSubmitUserId());
        taxReturnSubmission.setTaxReturn(this);
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
        taxReturnSubmissions.add(taxReturnSubmission);
        return taxReturnSubmission;
    }

    public void addOwner(User owner, boolean sync) {
        owners.add(owner);
        if (sync) owner.addTaxReturn(this, false);
    }

    public void addOwner(User owner) {
        this.addOwner(owner, true);
    }

    // Default to "package private"
    public static TaxReturn testObjectFactoryNoId() {
        TaxReturn taxReturn = new TaxReturn();
        taxReturn.createdAt = new Date();
        taxReturn.updatedAt = new Date();
        taxReturn.taxYear = 2024;
        taxReturn.facts = new HashMap<>();
        return taxReturn;
    }

    public static TaxReturn testObjectFactoryNoId(Date createdAt) {
        TaxReturn taxReturn = new TaxReturn();
        taxReturn.createdAt = createdAt;
        taxReturn.updatedAt = new Date();
        taxReturn.taxYear = 2024;
        taxReturn.facts = new HashMap<>();
        return taxReturn;
    }

    public static TaxReturn testObjectFactory() {
        TaxReturn taxReturn = testObjectFactoryNoId();
        taxReturn.createdAt = new Date();
        taxReturn.updatedAt = new Date();
        taxReturn.id = UUID.randomUUID();
        taxReturn.dataImportBehavior = DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2.name();
        return taxReturn;
    }

    public static TaxReturn fromTaxReturnTestObjectFactory(TaxReturn taxReturn) {
        // Populate the fields we care about.
        TaxReturn newTaxReturn = new TaxReturn();
        newTaxReturn.id = UUID.randomUUID();
        newTaxReturn.facts = taxReturn.facts;
        return newTaxReturn;
    }

    public boolean hasBeenSubmittedAtLeastOnce() {
        return this.getMostRecentSubmitTime() != null;
    }

    @Override
    public String toString() {
        return "TaxReturn{" + "taxYear=" + taxYear + " " + "taxReturnId=" + id + " " + "createdAt=" + createdAt + " "
                + '}';
    }
}
