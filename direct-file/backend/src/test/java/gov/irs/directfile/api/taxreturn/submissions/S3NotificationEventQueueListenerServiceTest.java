package gov.irs.directfile.api.taxreturn.submissions;

import com.amazon.sqs.javamessaging.message.SQSTextMessage;
import jakarta.jms.JMSException;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class S3NotificationEventQueueListenerServiceTest {

    S3NotificationEventQueueListenerService s3NotificationEventQueueListenerService;

    @Mock
    MessageQueueConfigurationProperties messageQueueConfigurationProperties;

    @Mock
    S3NotificationEventService s3NotificationEventService;

    String messageJson = """
                {"path": "adhoc_job.json"}
            """;

    @BeforeEach
    public void setup() {
        s3NotificationEventQueueListenerService = new S3NotificationEventQueueListenerService(
                messageQueueConfigurationProperties, s3NotificationEventService);
    }

    @SneakyThrows
    @Test
    void whenOnMessage_thenReadsRawText_thenCallsS3NotificationEventService() throws JMSException {
        SQSTextMessage message = new SQSTextMessage();
        message.setText(messageJson);
        s3NotificationEventQueueListenerService.onMessage(message);
        verify(s3NotificationEventService, times(1)).handleS3NotificationEvent(message.getText());
        verify(s3NotificationEventService, times(1)).handleS3NotificationEvent(messageJson);
    }
}
