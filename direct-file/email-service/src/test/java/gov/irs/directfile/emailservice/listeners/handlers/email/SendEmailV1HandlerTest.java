package gov.irs.directfile.emailservice.listeners.handlers.email;

import java.util.*;

import jakarta.mail.MessagingException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.services.EmailRecordKeepingService;
import gov.irs.directfile.emailservice.services.ISendService;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.email.SendEmailMessageVersion;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SendEmailV1HandlerTest {

    private SendEmailV1Handler handler;

    @Mock
    private EmailRecordKeepingService emailRecordKeepingService;

    private VersionedSendEmailMessage<AbstractSendEmailPayload> queueMessage;

    @Mock
    ISendService sendService;

    @BeforeEach
    public void setup() {
        handler = new SendEmailV1Handler(sendService, emailRecordKeepingService);

        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> emails = new LinkedHashMap<>();
        emails.put(
                HtmlTemplate.ACCEPTED,
                List.of(
                        new SendEmailQueueMessageBody(
                                "accept1@example.com",
                                "en",
                                UUID.randomUUID(),
                                "submissionId1",
                                UUID.randomUUID(),
                                UUID.fromString("00000000-0000-1111-1111-000000000000")),
                        new SendEmailQueueMessageBody(
                                "accept2@example.com",
                                "en",
                                UUID.randomUUID(),
                                "submissionId2",
                                UUID.randomUUID(),
                                null)));
        emails.put(
                HtmlTemplate.REJECTED,
                List.of(
                        new SendEmailQueueMessageBody(
                                "reject1@example.com",
                                "en",
                                UUID.randomUUID(),
                                "submissionId3",
                                UUID.randomUUID(),
                                null),
                        new SendEmailQueueMessageBody(
                                "reject2@example.com",
                                "en",
                                UUID.randomUUID(),
                                "submissionId4",
                                UUID.randomUUID(),
                                UUID.fromString("00000000-0000-2222-2222-000000000000"))));

        AbstractSendEmailPayload payload = new SendEmailPayloadV1(emails);
        queueMessage = new VersionedSendEmailMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, SendEmailMessageVersion.V1.getVersion()));
    }

    @Test
    void handleSendEmailMessage_success() throws Exception {
        doReturn(true).when(sendService).sendEmail(any());
        EmailProcessingResults emailProcessingResults = handler.handleSendEmailMessage(queueMessage);

        assertEquals(4, emailProcessingResults.countToSend());
        assertEquals(4, emailProcessingResults.countSent());
        assertEquals(0, emailProcessingResults.countMessagingException());
        assertTrue(emailProcessingResults.emailsToResend().isEmpty());
        assertEquals(4, emailProcessingResults.successfullySentEmails().size());
    }

    @Test
    void
            handleSendEmailMessage_createsSendEmailObjectsFromSQSPayloadAndCallsEmailRecordKeepingServiceToRecordEmailAttemptsAndResults()
                    throws Exception {
        doReturn(true).when(sendService).sendEmail(any());
        doReturn(false).when(sendService).sendEmail(argThat((SendEmail email) -> email.getRecipientEmailAddress()
                .equals("reject1@example.com")));
        EmailProcessingResults emailProcessingResults = handler.handleSendEmailMessage(queueMessage);

        ArgumentCaptor<List<SendEmail>> captor = ArgumentCaptor.forClass(List.class);
        verify(emailRecordKeepingService, times(1)).recordSendEmails(captor.capture());
        verify(emailRecordKeepingService, times(1)).recordSendEmailResults(eq(emailProcessingResults));

        // verify that they have IDs set.
        // any email that already has an ID should not be overwritten.
        // any email without an ID should have one assigned
        assertEquals(4, captor.getValue().size());
        assertEquals(
                UUID.fromString("00000000-0000-1111-1111-000000000000"),
                captor.getValue().get(0).getId());
        assertNotNull(captor.getValue().get(1).getId());
        assertNotNull(captor.getValue().get(2).getId());
        assertEquals(
                UUID.fromString("00000000-0000-2222-2222-000000000000"),
                captor.getValue().get(3).getId());
    }

    @Test
    void handleSendEmailMessage_failure() throws Exception {
        doReturn(false).when(sendService).sendEmail(any());
        EmailProcessingResults emailProcessingResults = handler.handleSendEmailMessage(queueMessage);

        assertEquals(4, emailProcessingResults.countToSend());
        assertEquals(0, emailProcessingResults.countSent());
        assertEquals(0, emailProcessingResults.countMessagingException());
        assertEquals(4, emailProcessingResults.emailsToResend().size());
        assertTrue(emailProcessingResults.successfullySentEmails().isEmpty());
    }

    @Test
    void handleSendEmailMessage_partial() throws Exception {
        doReturn(true).when(sendService).sendEmail(any());
        doReturn(false).when(sendService).sendEmail(argThat((SendEmail email) -> email.getRecipientEmailAddress()
                .equals("reject1@example.com")));
        EmailProcessingResults emailProcessingResults = handler.handleSendEmailMessage(queueMessage);

        assertEquals(4, emailProcessingResults.countToSend());
        assertEquals(3, emailProcessingResults.countSent());
        assertEquals(0, emailProcessingResults.countMessagingException());
        assertEquals(1, emailProcessingResults.emailsToResend().size());
        assertEquals(3, emailProcessingResults.successfullySentEmails().size());

        SendEmailPayloadV1 payload = (SendEmailPayloadV1) queueMessage.getPayload();
        SendEmailQueueMessageBody email =
                payload.getEmails().get(HtmlTemplate.REJECTED).getFirst();
        SendEmail firstSendEmail = emailProcessingResults.emailsToResend().getFirst();

        // Look at details of data structure to make sure it appears correct.
        assertEquals(email.getTo(), firstSendEmail.getRecipientEmailAddress());
        assertEquals(email.getLanguageCode(), firstSendEmail.getLanguageCode());
        assertEquals(email.getTaxReturnId(), firstSendEmail.getTaxReturnId());
        assertEquals(HtmlTemplate.REJECTED, firstSendEmail.getEmailType());
    }

    @Test
    void handleSendEmailMessage_partialException() throws Exception {
        doReturn(true).when(sendService).sendEmail(any());

        doThrow(MessagingException.class)
                .when(sendService)
                .sendEmail(argThat(
                        (SendEmail email) -> email.getRecipientEmailAddress().equals("accept2@example.com")));
        EmailProcessingResults emailProcessingResults = handler.handleSendEmailMessage(queueMessage);

        assertEquals(4, emailProcessingResults.countToSend());
        assertEquals(3, emailProcessingResults.countSent());
        assertEquals(1, emailProcessingResults.countMessagingException());
        assertEquals(0, emailProcessingResults.emailsToResend().size());
        assertEquals(3, emailProcessingResults.successfullySentEmails().size());
    }

    @Test
    void convertMessagesToSendEmails_processesMessageCorrectWithoutException() throws Exception {
        SendEmailPayloadV1 payload = (SendEmailPayloadV1) queueMessage.getPayload();

        List<SendEmail> emails = handler.convertMessagesToSendEmails(payload);
        assertEquals(emails.size(), 4);
        assertEquals(
                4,
                emails.stream()
                        .filter(email -> Objects.equals(email.getLanguageCode(), "en"))
                        .toList()
                        .size());
        assertEquals(
                2,
                emails.stream()
                        .filter(email -> email.getEmailType() == HtmlTemplate.REJECTED)
                        .toList()
                        .size());
        assertEquals(
                2,
                emails.stream()
                        .filter(email -> email.getEmailType() == HtmlTemplate.ACCEPTED)
                        .toList()
                        .size());
    }
}
