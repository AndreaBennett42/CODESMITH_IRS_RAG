package gov.irs.directfile.emailservice.listeners;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.amazon.sqs.javamessaging.message.SQSTextMessage;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.jms.JMSException;
import lombok.SneakyThrows;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlRequest;
import software.amazon.awssdk.services.sqs.model.GetQueueUrlResponse;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;

import gov.irs.directfile.emailservice.config.EmailServiceConfigurationProperties;
import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.services.EmailRecordKeepingService;
import gov.irs.directfile.emailservice.services.ISendService;
import gov.irs.directfile.emailservice.services.SqsConnectionSetupService;
import gov.irs.directfile.emailservice.services.TemplateService;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@SpringBootTest({"LOCAL_WRAPPING_KEY=9mteZFY+gIVfMFywgvpLpyVl+8UIcNoIWpGaHX4jDFU="})
@ActiveProfiles({"blackhole", "test"})
class SendEmailMessageQueueListenerTest {
    @Autowired
    EmailServiceConfigurationProperties configProps;

    SendEmailMessageQueueListener sendEmailMessageQueueListener;

    @MockBean
    SqsConnectionSetupService sqsConnectionSetupService;

    @MockBean
    ISendService sendService;

    @MockBean
    SqsClient sqsClient;

    @MockBean
    TemplateService templateService;

    @Mock
    EmailRecordKeepingService emailRecordKeepingService;

    @Mock
    GetQueueUrlResponse getQueueUrlResponse;

    private Logger listenerLogger;
    private ListAppender<ILoggingEvent> listAppender;

    String messageJson =
            """
                    {
                        "payload": {
                            "@type": "SendEmailPayloadV1",
                            "emails": {
                                "ACCEPTED": [
                                    {
                                        "to": "accept1@example.com",
                                        "languageCode": "en",
                                        "taxReturnId": "00000000-0000-1111-1111-000000000000",
                                        "context":{"someInt":15,"someBoolean":true,"someString":"hello"}
                                    },
                                    {
                                        "to": "accept2@example.com",
                                        "languageCode": "en",
                                        "taxReturnId": "00000000-0000-2222-2222-000000000000"
                                    }
                                ],
                                "REJECTED": [
                                    {
                                        "to": "reject1@example.com",
                                        "languageCode": "en",
                                        "taxReturnId": "00000000-0000-3333-3333-000000000000"
                                    },
                                    {
                                        "to": "reject2@example.com",
                                        "languageCode": "en",
                                        "taxReturnId": "00000000-0000-4444-4444-000000000000"
                                    }
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
        when(sqsClient.getQueueUrl(any(GetQueueUrlRequest.class))).thenReturn(getQueueUrlResponse);
        when(getQueueUrlResponse.queueUrl()).thenReturn("testQueryUrl");
        sendEmailMessageQueueListener =
                new SendEmailMessageQueueListener(configProps, sendService, sqsClient, emailRecordKeepingService);
        listenerLogger = (Logger) LoggerFactory.getLogger(SendEmailMessageQueueListener.class);
        listAppender = new ListAppender<>();
        listAppender.setContext((LoggerContext) LoggerFactory.getILoggerFactory());
        listAppender.start();
        listenerLogger.addAppender(listAppender);
    }

    @AfterEach
    public void tearDown() {
        listenerLogger.detachAppender(listAppender);
    }

    @SneakyThrows
    @Test
    void onMessage_ReadsTreeCorrectly_LogsCorrectLinesWhenAllEmailsAreSent() throws JMSException {
        SQSTextMessage message = new SQSTextMessage();
        message.setText(messageJson);
        doReturn(true).when(sendService).sendEmail(any());
        sendEmailMessageQueueListener.onMessage(message);
        List<ILoggingEvent> logsList = listAppender.list;
        ILoggingEvent firstEvent = logsList.get(0);
        ILoggingEvent secondEvent = logsList.get(1);
        assertEquals(Level.INFO, firstEvent.getLevel());
        assertEquals(Level.INFO, secondEvent.getLevel());
        assertEquals("Received a send mail request", firstEvent.getMessage());
        assertEquals("All emails sent", secondEvent.getMessage());
    }

    @SneakyThrows
    @Test
    void onMessage_ReadsTreeCorrectly_LogsCorrectLinesWhenAllEmailsAreNotSent() throws JMSException {
        SQSTextMessage message = new SQSTextMessage();
        message.setText(messageJson);
        doReturn(false).when(sendService).sendEmail(any());
        sendEmailMessageQueueListener.onMessage(message);
        List<ILoggingEvent> logsList = listAppender.list;
        ILoggingEvent firstEvent = logsList.get(0);
        ILoggingEvent secondEvent = logsList.get(1);
        assertEquals(Level.INFO, firstEvent.getLevel());
        assertEquals(Level.ERROR, secondEvent.getLevel());
        assertEquals("Received a send mail request", firstEvent.getMessage());
        assertEquals(
                String.format(
                        "Zero emails sent in batch size of %s, likely due to dropped connection to email relay. Returning and not acknowledging to force re-enqueue",
                        4),
                secondEvent.getMessage());
    }

    @SneakyThrows
    @Test
    public void onMessage_GivenAllMessagesSendSuccessfully_AcknowledgesMessageAndDoesNotRequeueAnotherOne() {
        SQSTextMessage message = new SQSTextMessage();
        message.setText(messageJson);
        doReturn(true).when(sendService).sendEmail(any());
        sendEmailMessageQueueListener.onMessage(message);

        verify(sqsClient, never()).sendMessage(any(SendMessageRequest.class));
    }

    @SneakyThrows
    @Test
    public void onMessage_GivenSomeMessagesFailToSend_RequeuesAMessageContainingFailedEmails() {
        ObjectMapper objectMapper = new ObjectMapper();
        SQSTextMessage message = new SQSTextMessage();
        message.setText(messageJson);
        when(sendService.sendEmail(any()))
                .thenReturn(false) // trigger the first email to fail
                .thenReturn(false) // trigger the second email to fail
                .thenReturn(true); // all remaining emails will be successful

        sendEmailMessageQueueListener.onMessage(message);

        ArgumentCaptor<SendMessageRequest> argumentCaptor = ArgumentCaptor.forClass(SendMessageRequest.class);
        verify(sqsClient, times(1)).sendMessage(argumentCaptor.capture());

        SendMessageRequest sendMessageRequest = argumentCaptor.getValue();
        VersionedSendEmailMessage<SendEmailPayloadV1> messagePayload =
                objectMapper.readValue(sendMessageRequest.messageBody(), new TypeReference<>() {});

        List<SendEmailQueueMessageBody> emailsToResend = messagePayload.getPayload().getEmails().values().stream()
                .flatMap(Collection::stream)
                .toList();
        assertEquals(2, emailsToResend.size());
        assertEquals(
                UUID.fromString("00000000-0000-1111-1111-000000000000"),
                emailsToResend.get(0).getTaxReturnId()); // this is the first email in the messageJson payload
        assertEquals(
                UUID.fromString("00000000-0000-2222-2222-000000000000"),
                emailsToResend.get(1).getTaxReturnId()); // this is the second email in the messageJson payload
    }

    @SneakyThrows
    @Test
    public void onMessage_GivenSomeAllFailToSend_RequeuesAMessageContainingFailedEmails() {
        ObjectMapper objectMapper = new ObjectMapper();
        SQSTextMessage message = new SQSTextMessage();
        message.setText(messageJson);
        when(sendService.sendEmail(any())).thenReturn(false);

        sendEmailMessageQueueListener.onMessage(message);

        ArgumentCaptor<SendMessageRequest> argumentCaptor = ArgumentCaptor.forClass(SendMessageRequest.class);
        verify(sqsClient, times(1)).sendMessage(argumentCaptor.capture());

        SendMessageRequest sendMessageRequest = argumentCaptor.getValue();
        VersionedSendEmailMessage<SendEmailPayloadV1> messagePayload =
                objectMapper.readValue(sendMessageRequest.messageBody(), new TypeReference<>() {});

        List<SendEmailQueueMessageBody> emailsToResend = messagePayload.getPayload().getEmails().values().stream()
                .flatMap(Collection::stream)
                .toList();
        assertEquals(4, emailsToResend.size());
        assertEquals(
                UUID.fromString("00000000-0000-1111-1111-000000000000"),
                emailsToResend.get(0).getTaxReturnId()); // this is the first email in the messageJson payload
        assertEquals(
                UUID.fromString("00000000-0000-2222-2222-000000000000"),
                emailsToResend.get(1).getTaxReturnId()); // this is the second email in the messageJson payload
        assertEquals(
                UUID.fromString("00000000-0000-3333-3333-000000000000"),
                emailsToResend.get(2).getTaxReturnId()); // this is the third email in the messageJson payload
        assertEquals(
                UUID.fromString("00000000-0000-4444-4444-000000000000"),
                emailsToResend.get(3).getTaxReturnId()); // this is the third email in the messageJson payload
    }

    @Test
    void onMessage_ReadsTreeCorrectly_MalformedMessageRaisesRuntimeExceptionButCaught() throws JMSException {
        SQSTextMessage mockMessage = mock(SQSTextMessage.class);
        when(mockMessage.getText())
                .thenReturn("""
                {"some_key":"some_val_without_a_closing_string}\s
                """);

        sendEmailMessageQueueListener.onMessage(mockMessage);

        // Verify that message.acknowledge() is not called
        verify(mockMessage, never()).acknowledge();
    }

    @Test
    void convertSendEmailsToSendMessageRequest_ConvertsWithoutException() {
        final List<SendEmail> sendEmails = List.of(
                new SendEmail(
                        UUID.randomUUID(),
                        "submissionId1",
                        UUID.randomUUID(),
                        "recipient1@test",
                        Map.of("key1", "value1"),
                        "en",
                        HtmlTemplate.SUBMITTED),
                new SendEmail(
                        UUID.randomUUID(),
                        "submissionId2",
                        UUID.randomUUID(),
                        "recipient2@test",
                        Map.of("key2", "value2"),
                        "en",
                        HtmlTemplate.SUBMITTED));
        try {
            sendEmailMessageQueueListener.convertSendEmailsToSendMessageRequest(sendEmails);
        } catch (JsonProcessingException e) {
            fail();
        }
    }
}
