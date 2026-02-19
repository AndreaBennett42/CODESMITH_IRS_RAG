package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("submit.sns")
@AllArgsConstructor
@Getter
public class SnsConfig {
    private final String endpoint;
    private final String submissionConfirmationTopicArn;
    private final boolean submissionConfirmationPublishEnabled;
    private final String region;
    private final Credentials credentials;

    @AllArgsConstructor
    @Getter
    public static class Credentials {
        private final String accessKey;
        private final String secretKey;
    }
}
