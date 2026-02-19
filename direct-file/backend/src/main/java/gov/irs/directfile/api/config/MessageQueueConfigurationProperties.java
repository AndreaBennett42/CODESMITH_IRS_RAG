package gov.irs.directfile.api.config;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.hibernate.validator.constraints.URL;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
// ConfigurationProperties expects prefix to be in lowercase throws exception if "direct-file.aws.messageQueue" is used
// for example
@ConfigurationProperties("direct-file.aws.messagequeue")
@Getter
@AllArgsConstructor
public class MessageQueueConfigurationProperties {

    @NotBlank
    @URL
    private final String endpoint;

    private final boolean sqsMessageSendingEnabled;

    @NotBlank
    private final String dispatchQueue;

    @NotBlank
    private final String dlqStatusChangeQueue;

    @NotBlank
    private final String dlqSubmissionConfirmationQueue;

    @NotBlank
    private final String dlqS3NotificationEventQueue;

    @NotBlank
    private final String sendEmailQueue;

    @NotBlank
    private final String statusChangeQueue;

    @NotBlank
    private final String submissionConfirmationQueue;

    @NotBlank
    private final String s3NotificationEventQueue;

    @NotBlank
    private final String dataImportRequestQueue;

    @NotBlank
    private final String dlqDataImportRequestQueue;

    @NotBlank
    private final String dataImportResultQueue;

    @NotBlank
    private final String dlqDataImportResultQueue;
}
