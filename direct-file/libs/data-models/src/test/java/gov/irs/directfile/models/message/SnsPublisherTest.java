package gov.irs.directfile.models.message;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.http.SdkHttpResponse;
import software.amazon.awssdk.services.sns.SnsClient;
import software.amazon.awssdk.services.sns.model.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SnsPublisherTest {
    private SnsPublisher snsPublisher;

    @Mock
    private SnsClient snsClient;

    @Mock
    private PublishResponse publishResponse;

    @Mock
    private SdkHttpResponse sdkHttpResponseForPublish;

    private final String topicArn = "arn:aws:sns:us-west-2:000000000000:some-topic";
    private final String message = "hello";

    @BeforeEach
    public void setup() {
        snsPublisher = new SnsPublisher(snsClient, topicArn);
    }

    @Test
    public void publish_success() {
        when(snsClient.publish(any(PublishRequest.class))).thenReturn(publishResponse);
        when(publishResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForPublish);
        when(sdkHttpResponseForPublish.isSuccessful()).thenReturn(true);

        snsPublisher.publish(message);

        PublishRequest publishRequest =
                PublishRequest.builder().topicArn(topicArn).message(message).build();

        ArgumentCaptor<PublishRequest> publishArgumentCaptor = ArgumentCaptor.forClass(PublishRequest.class);
        verify(snsClient, times(1)).publish(publishArgumentCaptor.capture());
        assertEquals(publishRequest, publishArgumentCaptor.getValue());
    }

    @Test
    public void publish_publishFails() {
        when(snsClient.publish(any(PublishRequest.class)))
                .thenThrow(NotFoundException.builder().build());

        assertThrows(PublisherException.class, () -> snsPublisher.publish(message));
        verify(snsClient, times(1)).publish(any(PublishRequest.class));
    }

    @Test
    public void publish_publishResponseFails() {
        when(snsClient.publish(any(PublishRequest.class))).thenReturn(publishResponse);
        when(publishResponse.sdkHttpResponse()).thenReturn(sdkHttpResponseForPublish);
        when(sdkHttpResponseForPublish.isSuccessful()).thenReturn(false);

        assertThrows(PublisherException.class, () -> snsPublisher.publish(message));
        verify(snsClient, times(1)).publish(any(PublishRequest.class));
    }
}
