package gov.irs.directfile.api.taxreturn.models;

import java.util.*;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import gov.irs.directfile.api.taxreturn.dto.Status;
import gov.irs.directfile.models.message.SubmissionEventFailureCategoryEnum;
import gov.irs.directfile.models.message.SubmissionEventFailureDetailEnum;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Getter
@Entity
@Table(name = "submission_events")
public class SubmissionEvent {
    @Id
    @GeneratedValue(generator = "UUID4")
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Setter
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "taxreturn_submission_id")
    private TaxReturnSubmission submission;

    @Setter
    @Column(nullable = false, updatable = false, name = "created_at")
    private Date createdAt;

    @Getter(AccessLevel.NONE)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "event_type")
    private String eventType;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "failure_category")
    private String failureCategory;

    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(columnDefinition = "varchar", name = "failure_detail")
    private String failureDetail;

    public SubmissionEventTypeEnum getEventType() {
        return SubmissionEventTypeEnum.getEnum(eventType);
    }

    public void setEventType(SubmissionEventTypeEnum submissionEventType) {
        this.eventType = submissionEventType.getEventType().toLowerCase();
    }

    public SubmissionEventFailureCategoryEnum getFailureCategory() {
        return SubmissionEventFailureCategoryEnum.getEnum(failureCategory.toUpperCase());
    }

    public void setFailureCategory(String failureCategory) {
        this.failureCategory = failureCategory.toLowerCase();
    }

    public void setFailureCategory(SubmissionEventFailureCategoryEnum failureCategoryEnum) {
        this.failureCategory = failureCategoryEnum.getFailureCategory().toLowerCase();
    }

    public SubmissionEventFailureDetailEnum getFailureDetail() {
        return SubmissionEventFailureDetailEnum.getEnum(failureDetail.toUpperCase());
    }

    public void setFailureDetail(String failureDetail) {
        this.failureDetail = failureDetail.toLowerCase();
    }

    public void setFailureDetail(SubmissionEventFailureDetailEnum failureDetailEnum) {
        this.failureDetail = failureDetailEnum.getFailureDetail().toLowerCase();
    }

    public Status getStatus() {
        switch (this.getEventType()) {
            case ACCEPTED:
                return Status.Accepted;
            case REJECTED:
                return Status.Rejected;
            default:
                return Status.Pending;
        }
    }

    public static SubmissionEvent testObjectFactory() {
        return testObjectFactory(TaxReturnSubmission.testObjectFactory());
    }

    public static SubmissionEvent testObjectFactory(
            TaxReturnSubmission taxReturnSubmission, SubmissionEventTypeEnum submissionEventType) {
        SubmissionEvent submissionEvent = testObjectFactory(taxReturnSubmission);
        submissionEvent.setEventType(submissionEventType);
        return submissionEvent;
    }

    public static SubmissionEvent testObjectFactory(SubmissionEventTypeEnum submissionEventType) {
        SubmissionEvent submissionEvent = testObjectFactory();
        submissionEvent.setEventType(submissionEventType);
        return submissionEvent;
    }

    public static SubmissionEvent testObjectFactory(TaxReturnSubmission taxReturnSubmission) {
        SubmissionEvent submissionEvent = baseTestObjectFactory();
        submissionEvent.setSubmission(taxReturnSubmission);
        return submissionEvent;
    }

    private static SubmissionEvent baseTestObjectFactory() {
        SubmissionEvent submissionEvent = new SubmissionEvent();
        return submissionEvent;
    }
}
