package gov.irs.directfile.emailservice.domain;

import java.util.Date;
import java.util.Map;
import java.util.UUID;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import gov.irs.directfile.models.email.HtmlTemplate;

@Entity
@Table(name = "send_emails")
@Getter
@Setter
@NoArgsConstructor
public class SendEmail {
    @Id
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Column(name = "created_at", nullable = false, updatable = false)
    @CreationTimestamp
    private Date createdAt;

    @Column(name = "updated_at")
    @UpdateTimestamp
    private Date updatedAt;

    @Column(name = "tax_return_id", updatable = false)
    private UUID taxReturnId;

    @Column(name = "submission_id", updatable = false)
    private String submissionId;

    @Column(name = "user_id", updatable = false)
    private UUID userId;

    @Transient
    private String recipientEmailAddress;

    // Key/value pairs passed to a template engine for composing a dynamic message. Can be null.
    @Transient
    private Map<String, Object> context;

    @Column(name = "language_code", updatable = false)
    private String languageCode;

    @Enumerated(EnumType.STRING)
    @Column(name = "email_type", updatable = false)
    private HtmlTemplate emailType;

    public SendEmail(
            UUID taxReturnId,
            String submissionId,
            UUID userId,
            String recipientEmailAddress,
            Map<String, Object> context,
            String languageCode,
            HtmlTemplate emailType) {
        this.taxReturnId = taxReturnId;
        this.submissionId = submissionId;
        this.userId = userId;
        this.recipientEmailAddress = recipientEmailAddress;
        this.context = context;
        this.languageCode = languageCode;
        this.emailType = emailType;
    }
}
