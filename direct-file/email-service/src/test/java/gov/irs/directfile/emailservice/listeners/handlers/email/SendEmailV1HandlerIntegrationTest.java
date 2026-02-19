package gov.irs.directfile.emailservice.listeners.handlers.email;

import java.util.*;

import ch.qos.logback.classic.Level;
import com.amazonaws.encryptionsdk.CryptoMaterialsManager;
import org.assertj.core.util.Sets;
import org.hibernate.engine.jdbc.batch.JdbcBatchLogging;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import software.amazon.awssdk.services.sqs.SqsClient;

import gov.irs.directfile.emailservice.config.SqsClientConfiguration;
import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.domain.SendEmailResult;
import gov.irs.directfile.emailservice.extension.LoggerExtension;
import gov.irs.directfile.emailservice.repositories.SendEmailRepository;
import gov.irs.directfile.emailservice.repositories.SendEmailResultRepository;
import gov.irs.directfile.emailservice.services.SqsConnectionSetupService;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.encryption.DataEncryptDecrypt;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.email.SendEmailMessageVersion;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest({"LOCAL_WRAPPING_KEY=9mteZFY+gIVfMFywgvpLpyVl+8UIcNoIWpGaHX4jDFU="})
@ActiveProfiles({"blackhole", "test"})
class SendEmailV1HandlerIntegrationTest {
    @Autowired
    SendEmailV1Handler sendEmailV1Handler;

    @Autowired
    SendEmailRepository sendEmailRepository;

    @Autowired
    SendEmailResultRepository sendEmailResultRepository;

    @MockBean
    SqsClient sqsClient;

    @MockBean
    SqsClientConfiguration sqsClientConfiguration;

    @MockBean
    CryptoMaterialsManager cryptoMaterialsManager;

    @MockBean
    DataEncryptDecrypt dataEncryptDecrypt;

    @MockBean
    SqsConnectionSetupService sqsConnectionSetupService;

    @RegisterExtension
    private static final LoggerExtension batchLogVerifier = new LoggerExtension(Level.TRACE, JdbcBatchLogging.NAME);

    @Value("${spring.jpa.properties.hibernate.jdbc.batch_size}")
    private int batchSize;

    private VersionedSendEmailMessage<AbstractSendEmailPayload> queueMessage;

    private SendEmailQueueMessageBody sendEmailQueueMessageBody1 = new SendEmailQueueMessageBody(
            "accept1@example.com",
            "en",
            UUID.randomUUID(),
            "submissionId1",
            UUID.randomUUID(),
            UUID.fromString("00000000-0000-1111-1111-000000000000"));

    private SendEmailQueueMessageBody sendEmailQueueMessageBody2 = new SendEmailQueueMessageBody(
            "accept2@example.com", "en", UUID.randomUUID(), "submissionId2", UUID.randomUUID(), null);

    private SendEmailQueueMessageBody sendEmailQueueMessageBody3 = new SendEmailQueueMessageBody(
            "reject1@example.com", "en", UUID.randomUUID(), "submissionId3", UUID.randomUUID(), null);

    private SendEmailQueueMessageBody sendEmailQueueMessageBody4 = new SendEmailQueueMessageBody(
            "reject2@example.com",
            "en",
            UUID.randomUUID(),
            "submissionId4",
            UUID.randomUUID(),
            UUID.fromString("00000000-0000-2222-2222-000000000000"));

    private SendEmailQueueMessageBody sendEmailQueueMessageBody5 = new SendEmailQueueMessageBody(
            "noncompletion@example.com",
            "en",
            UUID.randomUUID(),
            "submissionId5",
            UUID.randomUUID(),
            null,
            Map.of("key1", "value1"));

    @BeforeEach
    public void setup() {
        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> emails = new LinkedHashMap<>();
        emails.put(HtmlTemplate.ACCEPTED, List.of(sendEmailQueueMessageBody1, sendEmailQueueMessageBody2));
        emails.put(HtmlTemplate.REJECTED, List.of(sendEmailQueueMessageBody3, sendEmailQueueMessageBody4));
        emails.put(HtmlTemplate.NON_COMPLETION_SURVEY, List.of(sendEmailQueueMessageBody5));

        AbstractSendEmailPayload payload = new SendEmailPayloadV1(emails);
        queueMessage = new VersionedSendEmailMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, SendEmailMessageVersion.V1.getVersion()));

        sendEmailResultRepository.deleteAll();
        sendEmailRepository.deleteAll();
    }

    @Test
    public void
            givenEmailsWithNoEmailIds_whenHandleSendEmailMessage_persistsSendEmailRecords_processesEmails_persistsSendEmailResults() {
        // simulated actual use case. When an SQS message is received and some of the emails already have IDs, we would
        // expect that these are already persisted
        List<SendEmail> prePersisted =
                sendEmailV1Handler.convertMessagesToSendEmails((SendEmailPayloadV1) queueMessage.getPayload()).stream()
                        .filter(sendEmail -> sendEmail.getId().toString().startsWith("00000"))
                        .toList();
        sendEmailRepository.saveAll(prePersisted);

        EmailProcessingResults emailProcessingResults = sendEmailV1Handler.handleSendEmailMessage(queueMessage);

        // verify that the emails were successfully sent (BlackholeSendService stubs all attempts to send an email as
        // successful)
        assertTrue(emailProcessingResults.emailsToResend().isEmpty());
        assertEquals(5, emailProcessingResults.successfullySentEmails().size());
        assertEquals(
                sendEmailQueueMessageBody1.getTaxReturnId(),
                emailProcessingResults.successfullySentEmails().get(0).getTaxReturnId());
        assertEquals(
                sendEmailQueueMessageBody2.getTaxReturnId(),
                emailProcessingResults.successfullySentEmails().get(1).getTaxReturnId());
        assertEquals(
                sendEmailQueueMessageBody3.getTaxReturnId(),
                emailProcessingResults.successfullySentEmails().get(2).getTaxReturnId());
        assertEquals(
                sendEmailQueueMessageBody4.getTaxReturnId(),
                emailProcessingResults.successfullySentEmails().get(3).getTaxReturnId());
        assertEquals(
                sendEmailQueueMessageBody5.getTaxReturnId(),
                emailProcessingResults.successfullySentEmails().get(4).getTaxReturnId());

        // building a hash map of taxReturnId to EmailQueueMessageBody for easier assertions
        HashMap<UUID, SendEmailQueueMessageBody> taxReturnIdToSendEmailQueueMessageBodyMap = new HashMap<>();
        taxReturnIdToSendEmailQueueMessageBodyMap.put(
                sendEmailQueueMessageBody1.getTaxReturnId(), sendEmailQueueMessageBody1);
        taxReturnIdToSendEmailQueueMessageBodyMap.put(
                sendEmailQueueMessageBody2.getTaxReturnId(), sendEmailQueueMessageBody2);
        taxReturnIdToSendEmailQueueMessageBodyMap.put(
                sendEmailQueueMessageBody3.getTaxReturnId(), sendEmailQueueMessageBody3);
        taxReturnIdToSendEmailQueueMessageBodyMap.put(
                sendEmailQueueMessageBody4.getTaxReturnId(), sendEmailQueueMessageBody4);
        taxReturnIdToSendEmailQueueMessageBodyMap.put(
                sendEmailQueueMessageBody5.getTaxReturnId(), sendEmailQueueMessageBody5);

        // build a set containing the IDs of SendEmail records in order to verify they are referenced in SendEmailResult
        // records.
        Set<UUID> sendEmailIds = new HashSet<>();

        List<SendEmail> persistedSendEmails = (List<SendEmail>) sendEmailRepository.findAll();
        assertEquals(5, persistedSendEmails.size());
        persistedSendEmails.stream().forEach(sendEmail -> {
            // verify that every SendEmail that did NOT already have an emailId from the SQS
            assertNotNull(sendEmail.getId());

            // recipientEmailAddress and context should NOT be persisted in database.
            assertNull(sendEmail.getRecipientEmailAddress());
            assertNull(sendEmail.getContext());
            // They are marked as @Transient
            // payload
            // should get one assigned to it
            sendEmailIds.add(sendEmail.getId()); // populate the set containing the IDs of SendEmail records

            // verify that SendEmails with existing emailIds did NOT have their emailIds overwritten
            if (taxReturnIdToSendEmailQueueMessageBodyMap
                            .get(sendEmail.getTaxReturnId())
                            .getEmailId()
                    != null) {
                assertEquals(
                        sendEmail.getId(),
                        taxReturnIdToSendEmailQueueMessageBodyMap
                                .get(sendEmail.getTaxReturnId())
                                .getEmailId());
            }
        });

        List<SendEmailResult> persistedSendEmailResults = (List<SendEmailResult>) sendEmailResultRepository.findAll();
        assertEquals(5, persistedSendEmailResults.size());
        persistedSendEmailResults.stream().forEach(sendEmailResult -> {
            // every SendEmail that did not already have an emailId from the SQS payload should get one assigned to it
            assertNotNull(sendEmailResult.getSendEmail().getId());

            // verify that SendEmailResult records contain a foreign key reference to a SendEmail record
            assertTrue(sendEmailIds.contains(sendEmailResult.getSendEmail().getId()));

            // remove the keys one by one so we can verify that one SendEmailResult maps to 1 SendEmail
            sendEmailIds.remove(sendEmailResult.getSendEmail().getId());
        });
    }

    @Test
    public void oneToManyRelationshipBetweenSendEmailAndSendEmailResult() {
        SendEmail sendEmail = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "person1@aol.com",
                null,
                "en",
                HtmlTemplate.SUBMITTED);
        sendEmail.setId(UUID.randomUUID());

        SendEmailResult sendEmailResult = new SendEmailResult(sendEmail, true);

        sendEmailRepository.save(sendEmail);
        sendEmailResultRepository.save(sendEmailResult);

        List<SendEmailResult> result = (List<SendEmailResult>) sendEmailResultRepository.findAll();
        assertEquals(1, result.size());
        assertEquals(sendEmail.getId(), result.get(0).getSendEmail().getId());
    }

    @Test
    public void verifyDatabaseBatchingIsEnabled() {
        SendEmail sendEmail1 = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "person1@aol.com",
                null,
                "en",
                HtmlTemplate.SUBMITTED);
        sendEmail1.setId(UUID.randomUUID());
        SendEmail sendEmail2 = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "person2@aol.com",
                null,
                "en",
                HtmlTemplate.SUBMITTED);
        sendEmail2.setId(UUID.randomUUID());
        List<SendEmail> sendEmails = List.of(sendEmail1, sendEmail2);

        SendEmailResult sendEmailResult1 = new SendEmailResult(sendEmail1, true);
        SendEmailResult sendEmailResult2 = new SendEmailResult(sendEmail2, false);
        SendEmailResult sendEmailResult3 = new SendEmailResult(sendEmail2, true);
        List<SendEmailResult> sendEmailResults = List.of(sendEmailResult1, sendEmailResult2, sendEmailResult3);

        sendEmailRepository.saveAll(List.of(sendEmail1, sendEmail2));
        sendEmailResultRepository.saveAll(sendEmailResults);

        String expectedBatchLogMessage1 = String.format(
                "Executing JDBC batch (%s / %s) - `gov.irs.directfile.emailservice.domain.SendEmail#INSERT`",
                sendEmails.size(), batchSize);
        String expectedBatchLogMessage2 = String.format(
                "Executing JDBC batch (%s / %s) - `gov.irs.directfile.emailservice.domain.SendEmailResult#INSERT`",
                sendEmailResults.size(), batchSize);

        Set<String> expectedLogMessages = Sets.set(expectedBatchLogMessage1, expectedBatchLogMessage2);
        assertTrue(expectedMessagesWereLogged(expectedLogMessages, batchLogVerifier));
    }

    private boolean expectedMessagesWereLogged(Set<String> expectedLogMessages, LoggerExtension loggerExtension) {
        for (var loggingEvent : loggerExtension.getLoggingEvents()) {
            expectedLogMessages.remove(loggingEvent.getMessage());
        }

        return expectedLogMessages.isEmpty();
    }
}
