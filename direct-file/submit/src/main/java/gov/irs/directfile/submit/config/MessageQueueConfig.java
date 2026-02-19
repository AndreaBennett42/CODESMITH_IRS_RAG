package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("submit.messagequeue")
@AllArgsConstructor
@Getter
public class MessageQueueConfig {
    private final String endpoint;
    private final String dispatchQueue;
    private final String dlqDispatchQueue;
    private final String pendingSubmissionQueue;
    private final boolean pendingSubmissionPublishEnabled;
    private final String submissionConfirmationQueue;
    private final boolean submissionConfirmationPublishEnabled;
    private final boolean sqsMessageHandlingEnabled;
    private String region;
    private final Credentials credentials;

    @AllArgsConstructor
    @Getter
    public static class Credentials {
        private final String accessKey;
        private final String secretKey;
    }
}
