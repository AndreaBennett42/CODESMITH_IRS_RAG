package gov.irs.directfile.emailservice.services;

import java.util.Date;
import java.util.Locale;
import java.util.Map;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.mail.MailException;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import org.thymeleaf.context.Context;
import org.thymeleaf.context.IContext;
import org.thymeleaf.spring6.SpringTemplateEngine;
import org.thymeleaf.templateresolver.StringTemplateResolver;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.audit.LogUtil;
import gov.irs.directfile.emailservice.config.EmailServiceConfigurationProperties;
import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.domain.Template;
import gov.irs.directfile.emailservice.exceptions.MissingTemplateException;

@Service
@Slf4j
@Profile("send-email")
@SuppressFBWarnings(value = "CT_CONSTRUCTOR_THROW", justification = "Java 21 update")
@SuppressWarnings("PMD.MissingOverride")
public class SendService implements ISendService {
    // TODO: rate limiting?

    private final TemplateService templateService;

    private final JavaMailSender emailSender;

    private final SpringTemplateEngine templateEngine;

    private final String from;

    private final Resource logo;

    private final LogUtil structuredLogger;

    public SendService(
            EmailServiceConfigurationProperties config,
            TemplateService templateService,
            JavaMailSender emailSender,
            SpringTemplateEngine templateEngine,
            ResourceLoader resourceLoader) {
        from = config.getSender().getFrom();
        this.templateService = templateService;
        this.emailSender = emailSender;

        this.templateEngine = templateEngine;
        this.templateEngine.setTemplateResolver(new StringTemplateResolver()); // Resolver defaults to HTML mode

        logo = resourceLoader.getResource("classpath:templates/irs-logo-black.png");
        if (logo.isReadable()) {
            log.info("Loaded IRS logo from resources");
        } else {
            log.error("Failed to load IRS logo from resources");
            throw new RuntimeException("Failed to load IRS logo from resources");
        }
        this.structuredLogger = new LogUtil(log);
    }

    public boolean sendEmail(SendEmail email) throws MessagingException {
        Map<AuditLogElement, Object> logProperties = Map.of(
                AuditLogElement.taxReturnId,
                email.getTaxReturnId(),
                AuditLogElement.email,
                email.getRecipientEmailAddress());
        Date now = new Date();
        MimeMessage message = emailSender.createMimeMessage();
        try {
            populateMessage(message, email, now);
        } catch (MessagingException ex) {
            structuredLogger.logAtError("Error building email, see stacktrace for more details", logProperties, ex);
            throw ex;
        }

        try {
            structuredLogger.logAtInfo("Attempting to send email for template " + email.getEmailType(), logProperties);
            emailSender.send(message);
            structuredLogger.logAtInfo(email.getEmailType() + " email sent", logProperties);
            return true;
        } catch (MailException ex) {
            structuredLogger.logAtError("Unable to send email for template " + email.getEmailType(), logProperties, ex);
            return false;
        }
    }

    void populateMessage(MimeMessage message, SendEmail email, Date sentDate) throws MessagingException {

        final Template template;
        try {
            template = templateService.getTemplate(email.getLanguageCode(), email.getEmailType());
        } catch (MissingTemplateException e) {
            log.error(String.format("Bad template request: %s/%s", email.getLanguageCode(), email.getEmailType()));
            throw new RuntimeException(e);
        }

        String body;
        if (template.isDynamic()) {
            IContext context = new Context(Locale.of(email.getLanguageCode()), email.getContext());
            body = templateEngine.process(template.getBody(), context);
        } else {
            body = template.getBody();
        }

        MimeMessageHelper helper = new MimeMessageHelper(message, true);
        helper.setTo(email.getRecipientEmailAddress());
        helper.setFrom(from);
        helper.setSubject(template.getSubject());
        helper.setText(body, true);
        helper.setSentDate(sentDate);
        helper.addInline("Logo", logo);
    }
}
