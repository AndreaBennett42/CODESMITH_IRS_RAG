package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sns.SnsClient;

import gov.irs.directfile.api.config.SnsConfigurationProperties;
import gov.irs.directfile.models.message.SnsPublisher;

@Service
@ConditionalOnProperty(value = "direct-file.aws.sns.submission-confirmation-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(SnsConfigurationProperties.class)
public class SubmissionConfirmationSnsPublisher extends SnsPublisher implements SubmissionConfirmationPublisher {
    public SubmissionConfirmationSnsPublisher(
            SnsClient snsClient, SnsConfigurationProperties snsConfigurationProperties) {
        super(snsClient, snsConfigurationProperties.getSubmissionConfirmationTopicArn());
    }
}
