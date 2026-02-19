package gov.irs.directfile.status.services;

import com.amazon.sqs.javamessaging.message.SQSTextMessage;
import jakarta.jms.JMSException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.status.config.MessageQueueConfiguration;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MessageQueueListenerServiceTest {
    @Mock
    private MessageQueueConfiguration messageQueueConfiguration;

    @Mock
    private PendingSubmissionMessageRouter pendingSubmissionMessageRouter;

    @Mock
    private SubmissionConfirmationMessageRouter submissionConfirmationMessageRouter;

    private MessageQueueListenerService messageQueueListenerService;

    String pendingSubmissionMessageJson =
            """
                {
                  "payload": {
                    "@type": "PendingSubmissionPayloadV1",
                    "pendings": [
                      {
                        "taxReturnId": "00000000-0000-1111-1111-000000000000",
                        "submissionId": "11111111"
                      },
                      {
                        "taxReturnId": "00000000-0000-2222-2222-000000000000",
                        "submissionId": "22222222"
                      }
                    ]
                  },
                  "headers": {
                    "headers": {
                      "VERSION": "1.0"
                    }
                  }
                }
            """;

    String submissionConfirmationMessageJson =
            """
                {
                  "payload": {
                    "@type": "SubmissionConfirmationPayloadV1",
                    "receipts": [
                      {
                        "taxReturnId": "f6cdd8d9-2606-4acc-a331-28554f9bc72b",
                        "submissionId": "submissionId1",
                        "receiptId": "receiptId1",
                        "submissionReceivedAt": 1721754334033
                      },
                      {
                        "taxReturnId": "248fcc48-b362-497e-8927-e3c88b653009",
                        "submissionId": "submissionId2",
                        "receiptId": "receiptId2",
                        "submissionReceivedAt": 1721754334033
                      }
                    ]
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
        messageQueueListenerService = new MessageQueueListenerService(
                messageQueueConfiguration, pendingSubmissionMessageRouter, submissionConfirmationMessageRouter);
    }

    @Test
    public void onMessage_success_pendingSubmissionMessage() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText()).thenReturn(pendingSubmissionMessageJson);

        assertDoesNotThrow(() -> {
            messageQueueListenerService.onMessage(mockMessage);

            // Verify that only the pending submission handler is called and message acknowledged
            verify(pendingSubmissionMessageRouter, times(1)).handlePendingSubmissionMessage(any());
            verify(submissionConfirmationMessageRouter, never()).handleSubmissionConfirmationMessage(any());
            verify(mockMessage, times(1)).acknowledge();
        });
    }

    @Test
    public void onMessage_success_submissionConfirmationMessage() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText()).thenReturn(submissionConfirmationMessageJson);

        assertDoesNotThrow(() -> {
            messageQueueListenerService.onMessage(mockMessage);

            // Verify that only the submission confirmation handler is called and message acknowledged
            verify(pendingSubmissionMessageRouter, never()).handlePendingSubmissionMessage(any());
            verify(submissionConfirmationMessageRouter, times(1)).handleSubmissionConfirmationMessage(any());
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

        messageQueueListenerService.onMessage(mockMessage);

        // Verify that handlers and message.acknowledge() are not called
        verify(pendingSubmissionMessageRouter, never()).handlePendingSubmissionMessage(any());
        verify(submissionConfirmationMessageRouter, never()).handleSubmissionConfirmationMessage(any());
        verify(mockMessage, never()).acknowledge();
    }
}
