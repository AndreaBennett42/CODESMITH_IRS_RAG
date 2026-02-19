package gov.irs.directfile.submit.service;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sns.SnsClient;

import gov.irs.directfile.models.message.SnsPublisher;
import gov.irs.directfile.submit.config.SnsConfig;

@Service
@ConditionalOnProperty(value = "submit.sns.submission-confirmation-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(SnsConfig.class)
public class SubmissionConfirmationSnsPublisher extends SnsPublisher implements SubmissionConfirmationPublisher {
    public SubmissionConfirmationSnsPublisher(SnsClient snsClient, SnsConfig snsConfig) {
        super(snsClient, snsConfig.getSubmissionConfirmationTopicArn());
    }
}
