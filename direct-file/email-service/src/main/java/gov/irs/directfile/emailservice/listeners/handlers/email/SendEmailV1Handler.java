package gov.irs.directfile.emailservice.listeners.handlers.email;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import jakarta.mail.MessagingException;
import org.springframework.stereotype.Service;

import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.services.EmailRecordKeepingService;
import gov.irs.directfile.emailservice.services.ISendService;
import gov.irs.directfile.models.message.email.VersionedSendEmailMessage;
import gov.irs.directfile.models.message.email.payload.AbstractSendEmailPayload;
import gov.irs.directfile.models.message.email.payload.SendEmailPayloadV1;

@Service
public class SendEmailV1Handler implements SendEmailHandler {
    private final ISendService sender;

    private final EmailRecordKeepingService emailRecordKeepingService;

    public SendEmailV1Handler(ISendService sender, EmailRecordKeepingService emailRecordKeepingService) {
        this.sender = sender;
        this.emailRecordKeepingService = emailRecordKeepingService;
    }

    @Override
    public EmailProcessingResults handleSendEmailMessage(VersionedSendEmailMessage<AbstractSendEmailPayload> message) {
        SendEmailPayloadV1 payload = (SendEmailPayloadV1) message.getPayload();

        List<SendEmail> emailsToSend = convertMessagesToSendEmails(payload);
        emailRecordKeepingService.recordSendEmails(emailsToSend);

        List<SendEmail> emailsToResend = new ArrayList<>();
        List<SendEmail> successfullySentEmails = new ArrayList<>();
        AtomicInteger countSent = new AtomicInteger(0);
        AtomicInteger countMessagingException = new AtomicInteger(0);
        emailsToSend.forEach(email -> {
            try {
                if (sender.sendEmail(email)) {
                    countSent.incrementAndGet();
                    successfullySentEmails.add(email);
                } else {
                    emailsToResend.add(email);
                }
            } catch (MessagingException e) {
                countMessagingException.incrementAndGet();
            }
        });

        EmailProcessingResults emailProcessingResults = new EmailProcessingResults(
                emailsToSend.size(),
                countSent.get(),
                countMessagingException.get(),
                emailsToResend,
                successfullySentEmails);

        emailRecordKeepingService.recordSendEmailResults(emailProcessingResults);

        return emailProcessingResults;
    }

    protected List<SendEmail> convertMessagesToSendEmails(SendEmailPayloadV1 payload) {
        ArrayList<SendEmail> emails = new ArrayList<>();

        for (var entry : payload.getEmails().entrySet()) {
            for (var email : entry.getValue()) {
                SendEmail sendEmail = new SendEmail(
                        email.getTaxReturnId(),
                        email.getSubmissionId(),
                        email.getUserId(),
                        email.getTo(),
                        email.getContext(),
                        email.getLanguageCode(),
                        entry.getKey() // HTMLTemplate
                        );

                // If an email from the payload has an emailId, then it's an email that the EmailService has already
                // seen, failed to confirm as sent to the recipient, and has re-queued via SQS.
                if (email.getEmailId() != null) {
                    sendEmail.setId(email.getEmailId());
                } else {
                    // if an email does not have an emailId, then this email has not previously been seen by the
                    // EmailService. This is a new email, so a new ID gets assigned to it.
                    sendEmail.setId(UUID.randomUUID());
                }

                emails.add(sendEmail);
            }
        }

        return emails;
    }
}
