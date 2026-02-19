package gov.irs.directfile.models.message;

import java.util.Map;
import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

@AllArgsConstructor
@Getter
@Setter
@NoArgsConstructor
@ToString
public class SendEmailQueueMessageBody {
    private String to;
    private String languageCode;
    private UUID taxReturnId;
    private String submissionId;
    private UUID userId;
    private UUID emailId;

    // Key/value pairs passed to a template engine for composing a dynamic message. Can be null.
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Map<String, Object> context;

    public SendEmailQueueMessageBody(String to, String langCode, UUID taxReturnId, String submissionId, UUID userId) {
        this.to = to;
        this.languageCode = langCode;
        this.taxReturnId = taxReturnId;
        this.submissionId = submissionId;
        this.userId = userId;
        this.emailId = null;
    }

    public SendEmailQueueMessageBody(
            String to, String langCode, UUID taxReturnId, String submissionId, UUID userId, UUID emailId) {
        this(to, langCode, taxReturnId, submissionId, userId);
        this.emailId = emailId;
    }
}
