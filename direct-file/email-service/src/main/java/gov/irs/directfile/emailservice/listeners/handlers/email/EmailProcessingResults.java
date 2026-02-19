package gov.irs.directfile.emailservice.listeners.handlers.email;

import java.util.List;

import gov.irs.directfile.emailservice.domain.SendEmail;

public record EmailProcessingResults(
        int countToSend,
        int countSent,
        int countMessagingException,
        List<SendEmail> emailsToResend,
        List<SendEmail> successfullySentEmails) {
    public EmailProcessingResults {
        // Make List immutable (a mutable List causes a linter error)
        if (emailsToResend == null) {
            emailsToResend = List.of();
        } else {
            emailsToResend = List.copyOf(emailsToResend);
        }
    }
}
