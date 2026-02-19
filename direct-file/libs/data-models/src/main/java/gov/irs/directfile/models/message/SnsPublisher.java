package gov.irs.directfile.models.message;

import java.util.List;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import software.amazon.awssdk.services.sns.SnsClient;
import software.amazon.awssdk.services.sns.model.*;

@Slf4j
public class SnsPublisher implements Publisher {
    protected final SnsClient snsClient;
    protected final String topicArn;

    public SnsPublisher(SnsClient snsClient, String topicArn) {
        this.snsClient = snsClient;
        this.topicArn = topicArn;
    }

    // At startup, attempt to look up the topic ARN and its subscriptions as a validation.
    @PostConstruct
    public void init() {
        ListSubscriptionsByTopicRequest listSubscriptionsByTopicRequest =
                ListSubscriptionsByTopicRequest.builder().topicArn(topicArn).build();
        ListSubscriptionsByTopicResponse listSubscriptionsByTopicResponse;
        try {
            listSubscriptionsByTopicResponse = snsClient.listSubscriptionsByTopic(listSubscriptionsByTopicRequest);
        } catch (Exception e) {
            String errorMessage = "Exception calling SNS listSubscriptionsByTopic: " + e.getMessage();
            log.error(errorMessage, e);
            throw new PublisherException(errorMessage, e);
        }
        log.info("SNS topic ARN found: {}", topicArn);

        List<Subscription> subscriptions = listSubscriptionsByTopicResponse.subscriptions();
        if (subscriptions.isEmpty()) {
            String errorMessage = "SNS subscription not found";
            log.error(errorMessage);
            throw new PublisherException(errorMessage);
        } else {
            int subscriptionCount = subscriptions.size();
            for (int i = 0; i < subscriptions.size(); i++) {
                log.info(
                        "SNS subscription {} of {} found: {}",
                        i + 1,
                        subscriptionCount,
                        subscriptions.get(i).toString());
            }
        }
    }

    @Override
    public void publish(String message) {
        PublishRequest publishRequest =
                PublishRequest.builder().topicArn(topicArn).message(message).build();
        PublishResponse publishResponse;
        try {
            publishResponse = snsClient.publish(publishRequest);
        } catch (Exception e) {
            String errorMessage = "Exception calling SNS publish: " + e.getMessage();
            log.error(errorMessage, e);
            throw new PublisherException(errorMessage, e);
        }

        if (publishResponse.sdkHttpResponse().isSuccessful()) {
            log.info("Published message to SNS topic ARN: {}", topicArn);
        } else {
            String errorMessage = "SNS publish request was unsuccessful. HTTP Status code: "
                    + publishResponse.sdkHttpResponse().statusCode();
            log.error(errorMessage);
            throw new PublisherException(errorMessage);
        }
    }
}
