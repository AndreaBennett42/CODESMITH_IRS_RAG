package gov.irs.directfile.api.taxreturn.models;

import java.util.*;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import gov.irs.directfile.models.FactTypeWithItem;
import gov.irs.directfile.models.message.SubmissionEventFailureCategoryEnum;
import gov.irs.directfile.models.message.SubmissionEventFailureDetailEnum;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

/*This table left outer joins (meaning one tax return to many tax return submissions) to the tax return table on the tax return id.

It contains
a sequence number that represents how many attempts at submitting this year we have made,
the MeF submission ID,
the original MeF submission ID,
the MeF status,
the reject reasons,
and a copy of the submitted fact graph.

 */
@Slf4j
@Getter
@Entity
@Table(name = "taxreturn_submissions")
@EntityListeners({TaxReturnEntityListener.class})
@SuppressWarnings("PMD.AvoidDuplicateLiterals")
public class TaxReturnSubmission implements TaxReturnEntity {
    // dirty data: this is never a valid encrypted value that can be store in the database,
    // so will always dirty the entity when written to a field
    private static final String DD = "DD-to-force-reencryption-by-entity-listener";

    public enum SubmissionType {
        ONLINE_FILER,
        ERO
    }

    @Id
    @Setter
    @GeneratedValue(generator = "UUID4")
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Setter
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "taxreturn_id")
    private TaxReturn taxReturn;

    @Setter
    @Column(name = "taxreturn_id", updatable = false, insertable = false)
    private UUID taxReturnId;

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

    @Setter
    @Column(nullable = false, updatable = false, name = "created_at")
    private Date createdAt;

    @Setter
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "submission_id")
    private String submissionId;

    @Setter
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "receipt_id")
    private String receiptId;

    @Setter
    @Column(name = "submission_received_at")
    private Date submissionReceivedAt;

    @Setter
    @JdbcTypeCode(SqlTypes.INTEGER)
    @Column(nullable = true, updatable = true, name = "submission_sequence_id")
    private Integer submissionSequenceId;

    @Setter
    @Column
    private UUID submitUserId;

    @Setter
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(nullable = false, name = "submission_type")
    private String submissionType = SubmissionType.ONLINE_FILER.name();

    @OneToMany(mappedBy = "submission", cascade = CascadeType.ALL, targetEntity = SubmissionEvent.class)
    private Set<SubmissionEvent> submissionEvents = new HashSet<>();

    public SubmissionEvent addSubmissionEvent(SubmissionEventTypeEnum eventType) {
        SubmissionEvent submissionEvent = new SubmissionEvent();
        submissionEvent.setEventType(eventType);
        submissionEvent.setSubmission(this);
        submissionEvent.setCreatedAt(new Date());
        submissionEvents.add(submissionEvent);
        log.info("Submission event added with eventType: {}", eventType);
        return submissionEvent;
    }

    public SubmissionEvent addSubmissionEventForTest(SubmissionEventTypeEnum eventType, Date date) {
        SubmissionEvent submissionEvent = new SubmissionEvent();
        submissionEvent.setEventType(eventType);
        submissionEvent.setSubmission(this);
        submissionEvent.setCreatedAt(date);
        submissionEvents.add(submissionEvent);
        log.info("Submission event added with eventType: {} and createdAt: {}", eventType, date);
        return submissionEvent;
    }

    public SubmissionEvent addSubmissionEvent(
            SubmissionEventTypeEnum eventType,
            SubmissionEventFailureCategoryEnum failureCategory,
            SubmissionEventFailureDetailEnum failureDetail) {
        SubmissionEvent submissionEvent = addSubmissionEvent(eventType);
        if (failureCategory != null) {
            submissionEvent.setFailureCategory(failureCategory);
        }
        if (failureDetail != null) {
            submissionEvent.setFailureDetail(failureDetail);
        }
        return submissionEvent;
    }

    public SubmissionEvent addSubmissionEvent(
            SubmissionEventTypeEnum eventType, String failureCategory, String failureDetail) {
        SubmissionEvent submissionEvent = addSubmissionEvent(eventType);
        if (failureCategory != null) {
            submissionEvent.setFailureCategory(failureCategory);
        }
        if (failureDetail != null) {
            submissionEvent.setFailureDetail(failureDetail);
        }
        return submissionEvent;
    }

    public static TaxReturnSubmission testObjectFactory() {
        return testObjectFactory(TaxReturn.testObjectFactory());
    }

    public static TaxReturnSubmission testObjectFactory(TaxReturn taxReturn) {
        TaxReturnSubmission taxReturnSubmission = baseTestObjectFactory();
        taxReturnSubmission.setTaxReturn(taxReturn);
        return taxReturnSubmission;
    }

    private static TaxReturnSubmission baseTestObjectFactory() {
        TaxReturnSubmission taxReturnSubmission = new TaxReturnSubmission();
        return taxReturnSubmission;
    }
}
