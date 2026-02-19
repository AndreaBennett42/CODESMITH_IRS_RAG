package gov.irs.directfile.api.taxreturn.submissions;

import com.amazon.sqs.javamessaging.message.SQSTextMessage;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.jms.JMSException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class StatusQueueListenerServiceTest {
    @Mock
    private StatusChangeMessageRouter statusChangeMessageRouter;

    @Mock
    MessageQueueConfigurationProperties messageQueueConfigurationProperties;

    private StatusQueueListenerService statusQueueListenerService;

    String messageJson =
            """
                {
                  "payload": {
                    "@type": "StatusChangePayloadV1",
                    "statusSubmissionIdMap": {
                       "accepted": [
                         "123456789"
                       ],
                       "rejected": [
                         "987654321"
                       ]
                     }
                  },
                  "headers": {
                    "headers": {
                      "VERSION": "1.0"
                    }
                  }
                }
            """;

    @BeforeEach
    public void setup() {
        statusQueueListenerService = new StatusQueueListenerService(
                messageQueueConfigurationProperties, statusChangeMessageRouter, new ObjectMapper());
    }

    @Test
    void onMessage_success() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText()).thenReturn(messageJson);

        assertDoesNotThrow(() -> {
            statusQueueListenerService.onMessage(mockMessage);
            verify(statusChangeMessageRouter, times(1)).handleStatusChangeMessage(any());
            verify(mockMessage, times(1)).acknowledge();
        });
    }

    @Test
    void onMessage_exceptionThrown() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText())
                .thenReturn(
                        """
                        {"some_key":"some_val_without_a_closing_string}\s
                        """);

        statusQueueListenerService.onMessage(mockMessage);

        // Verify that message.acknowledge() is not called
        verify(mockMessage, never()).acknowledge();
    }
}
