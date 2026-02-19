package gov.irs.directfile.emailservice.services;

import java.util.*;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.Mock;
import org.mockito.Spy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.core.io.ResourceLoader;
import org.springframework.mail.MailSendException;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.test.context.ActiveProfiles;
import org.thymeleaf.context.Context;
import org.thymeleaf.context.IContext;
import org.thymeleaf.spring6.SpringTemplateEngine;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.emailservice.config.EmailServiceConfigurationProperties;
import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.extension.LoggerExtension;
import gov.irs.directfile.models.email.HtmlTemplate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@SpringBootTest({"LOCAL_WRAPPING_KEY=9mteZFY+gIVfMFywgvpLpyVl+8UIcNoIWpGaHX4jDFU="})
@ActiveProfiles({"send-email", "test"})
class SendServiceTest {
    @RegisterExtension
    public static LoggerExtension loggerExtension = new LoggerExtension(Level.INFO, SendService.class.getName());

    @Autowired
    EmailServiceConfigurationProperties config;

    @Autowired
    ResourceLoader resourceLoader;

    @Autowired
    TemplateService templateService;

    @MockBean
    SqsConnectionSetupService sqsConnectionSetupService;

    @Mock
    JavaMailSender javaMailSender;

    @Spy
    SpringTemplateEngine springTemplateEngine;

    @Test
    void LoadsCidResources() {
        // SendService constructor will try to load the IRS logo and will log an error
        // if not successful

        new SendService(config, templateService, javaMailSender, springTemplateEngine, resourceLoader);
        assertEquals(1, loggerExtension.getLoggingEvents().size());
        assertEquals(
                "Loaded IRS logo from resources",
                loggerExtension.getLoggingEvents().get(0).getMessage());
    }

    @Test
    void PopulateMessageDoesNotThrowException() {
        MimeMessage message = mock(MimeMessage.class);

        SendService sendService =
                new SendService(config, templateService, javaMailSender, springTemplateEngine, resourceLoader);
        SendEmail sendEmail = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "test@to.test",
                null,
                "en",
                HtmlTemplate.SUBMITTED);

        try {
            sendService.populateMessage(message, sendEmail, new Date());
        } catch (MessagingException e) {
            fail("Caught unexpected MessagingException", e);
        }

        verify(springTemplateEngine, never()).process(anyString(), any());
    }

    @Test
    void givenSendEmailIsDynamic_whenPopulateMessageSucceeds_thenTemplateIsPopulated() {
        MimeMessage message = mock(MimeMessage.class);

        SendService sendService =
                new SendService(config, templateService, javaMailSender, springTemplateEngine, resourceLoader);

        String surveyUrl = "http://localhost/survey";
        Map<String, Object> contextMap = Map.of("surveyUrl", surveyUrl);
        String languageCode = "en";
        SendEmail sendEmail = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "test@to.test",
                contextMap,
                languageCode,
                HtmlTemplate.NON_COMPLETION_SURVEY);

        // Need to capture the body return value from templateEngine.process
        IContext context = new Context(Locale.of(languageCode), contextMap);
        ResultCaptor<String> resultCaptor = new ResultCaptor<>();
        doAnswer(resultCaptor).when(springTemplateEngine).process(anyString(), any());

        try {
            sendService.populateMessage(message, sendEmail, new Date());
        } catch (MessagingException e) {
            fail("Caught unexpected MessagingException", e);
        }

        verify(springTemplateEngine, times(1)).process(anyString(), any());
        String result = resultCaptor.getResult();
        assertTrue(result.contains(surveyUrl));
    }

    @SneakyThrows
    @Test
    void givenMessage_whenSendEmailSucceeds_thenLogsSuccessMessages() {
        SendService sendService =
                new SendService(config, templateService, javaMailSender, springTemplateEngine, resourceLoader);
        SendEmail sendEmail = new SendEmail(
                UUID.fromString("00000000-0000-0000-0000-000000000000"),
                "submissionId",
                UUID.randomUUID(),
                "test@to.test",
                null,
                "en",
                HtmlTemplate.SUBMITTED);

        MimeMessage message = mock(MimeMessage.class);
        doReturn(message).when(javaMailSender).createMimeMessage();

        sendService.sendEmail(sendEmail);

        List<ILoggingEvent> events = loggerExtension.getLoggingEvents();
        assertEquals(3, events.size());

        Map<String, String> expectedProperties = Map.of(
                AuditLogElement.email.toString(),
                "test@to.test",
                AuditLogElement.taxReturnId.toString(),
                "00000000-0000-0000-0000-000000000000",
                AuditLogElement.cyberOnly.toString(),
                "true");
        assertThat(events.get(1).getMessage()).isEqualTo("Attempting to send email for template SUBMITTED");
        loggerExtension.verifyKeyValuePairsEqual(events.get(1), expectedProperties);
        assertThat(events.get(2).getMessage()).startsWith("SUBMITTED email sent");
        loggerExtension.verifyKeyValuePairsEqual(events.get(2), expectedProperties);
    }

    @SneakyThrows
    @Test
    void givenMessage_whenSendEmailFails_thenLogsFailureMessages() {
        SendService sendService =
                new SendService(config, templateService, javaMailSender, springTemplateEngine, resourceLoader);

        SendEmail sendEmail = new SendEmail(
                UUID.fromString("00000000-0000-0000-0000-000000000000"),
                "submissionId",
                UUID.randomUUID(),
                "test@to.test",
                null,
                "en",
                HtmlTemplate.SUBMITTED);

        MimeMessage message = mock(MimeMessage.class);
        doReturn(message).when(javaMailSender).createMimeMessage();
        doThrow(MailSendException.class).when(javaMailSender).send(any(MimeMessage.class));

        sendService.sendEmail(sendEmail);

        List<ILoggingEvent> events = loggerExtension.getLoggingEvents();
        assertEquals(3, events.size());

        Map<String, String> expectedProperties = Map.of(
                AuditLogElement.email.toString(),
                "test@to.test",
                AuditLogElement.taxReturnId.toString(),
                "00000000-0000-0000-0000-000000000000",
                AuditLogElement.cyberOnly.toString(),
                "true");
        assertThat(events.get(1).getMessage()).isEqualTo("Attempting to send email for template SUBMITTED");
        loggerExtension.verifyKeyValuePairsEqual(events.get(1), expectedProperties);
        assertThat(events.get(2).getMessage()).isEqualTo("Unable to send email for template SUBMITTED");
        loggerExtension.verifyKeyValuePairsEqual(events.get(2), expectedProperties);
    }
}
