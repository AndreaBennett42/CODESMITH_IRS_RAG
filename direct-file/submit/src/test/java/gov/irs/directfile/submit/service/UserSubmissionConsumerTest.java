package gov.irs.directfile.submit.service;

import com.amazon.sqs.javamessaging.message.SQSTextMessage;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.jms.JMSException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserSubmissionConsumerTest {
    @Mock
    private DispatchMessageRouter dispatchMessageRouter;

    private UserSubmissionConsumer userSubmissionConsumer;

    String messageJson =
            """
                {
                  "payload": {
                    "@type": "DispatchPayloadV1",
                    "dispatch": {
                       "id": "63ac8695-48e8-4078-81cb-85798f9023f6",
                       "userId": "d44d647b-ce88-4db3-9e5b-b69fb005d6fe",
                       "taxReturnId": "e3633e53-ae12-46e6-95bb-54838ad7b0d4",
                       "pathToManifest": "",
                       "pathToUserContext": "",
                       "pathToSubmission": "",
                       "mefSubmissionId": "12345"
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
        userSubmissionConsumer = new UserSubmissionConsumer(dispatchMessageRouter, new ObjectMapper());
    }

    @Test
    public void onMessage_success() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText()).thenReturn(messageJson);

        assertDoesNotThrow(() -> {
            userSubmissionConsumer.onMessage(mockMessage);
            verify(dispatchMessageRouter, times(1)).handleDispatchMessage(any());
            verify(mockMessage, times(1)).acknowledge();
        });
    }

    @Test
    public void onMessage_exceptionThrown() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText())
                .thenReturn(
                        """
                        {"some_key":"some_val_without_a_closing_string}\s
                        """);

        userSubmissionConsumer.onMessage(mockMessage);

        // Verify that message.acknowledge() is not called
        verify(mockMessage, never()).acknowledge();
    }
}
