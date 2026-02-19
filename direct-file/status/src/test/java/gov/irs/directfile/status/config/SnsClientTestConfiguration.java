package gov.irs.directfile.status.config;

import java.util.ArrayList;
import java.util.List;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import software.amazon.awssdk.services.sns.SnsClient;
import software.amazon.awssdk.services.sns.model.ListSubscriptionsByTopicRequest;
import software.amazon.awssdk.services.sns.model.ListSubscriptionsByTopicResponse;
import software.amazon.awssdk.services.sns.model.Subscription;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@TestConfiguration
public class SnsClientTestConfiguration {
    @Bean
    @Primary
    public SnsClient testSnsClient() {
        ListSubscriptionsByTopicResponse listSubscriptionsByTopicResponse =
                mock(ListSubscriptionsByTopicResponse.class);
        List<Subscription> subscriptions = new ArrayList<>();
        subscriptions.add(mock(Subscription.class));
        when(listSubscriptionsByTopicResponse.subscriptions()).thenReturn(subscriptions);

        SnsClient snsClient = mock(SnsClient.class);
        when(snsClient.listSubscriptionsByTopic(any(ListSubscriptionsByTopicRequest.class)))
                .thenReturn(listSubscriptionsByTopicResponse);
        return snsClient;
    }
}
