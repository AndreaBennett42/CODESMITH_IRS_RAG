package gov.irs.directfile.status.services;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sns.SnsClient;

import gov.irs.directfile.models.message.SnsPublisher;
import gov.irs.directfile.status.config.SnsConfiguration;

@Service
@ConditionalOnProperty(value = "status.sns.status-change-publish-enabled", havingValue = "true")
@EnableConfigurationProperties(SnsConfiguration.class)
public class StatusChangeSnsPublisher extends SnsPublisher implements StatusChangePublisher {
    public StatusChangeSnsPublisher(SnsClient snsClient, SnsConfiguration snsConfiguration) {
        super(snsClient, snsConfiguration.getStatusChangeTopicArn());
    }
}
