package gov.irs.directfile.emailservice.domain;

import java.util.Date;
import java.util.UUID;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

@Entity
@Table(name = "send_email_results")
@Getter
@Setter
@NoArgsConstructor
public class SendEmailResult {
    @Id
    @GeneratedValue(generator = "UUID4")
    private UUID id;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "send_email_id")
    private SendEmail sendEmail;

    @Column(name = "created_at", nullable = false, updatable = false)
    @CreationTimestamp
    private Date createdAt;

    @Column(name = "is_sent", nullable = false, updatable = false)
    private boolean isSent;

    public SendEmailResult(SendEmail sendEmail, boolean isSent) {
        this.sendEmail = sendEmail;
        this.isSent = isSent;
    }
}
