package gov.irs.directfile.emailservice.services;

import java.util.ArrayList;
import java.util.List;

import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.domain.SendEmailResult;
import gov.irs.directfile.emailservice.listeners.handlers.email.EmailProcessingResults;
import gov.irs.directfile.emailservice.repositories.SendEmailRepository;
import gov.irs.directfile.emailservice.repositories.SendEmailResultRepository;

@Service
@AllArgsConstructor
public class EmailRecordKeepingService {
    SendEmailRepository sendEmailRepository;
    SendEmailResultRepository sendEmailResultRepository;

    public void recordSendEmails(List<SendEmail> sendEmails) {
        sendEmailRepository.saveAll(sendEmails);
    }

    public void recordSendEmailResults(EmailProcessingResults emailProcessingResults) {
        List<SendEmailResult> listOfSendEmailResults = createSendEmailResults(emailProcessingResults);
        sendEmailResultRepository.saveAll(listOfSendEmailResults);
    }

    private List<SendEmailResult> createSendEmailResults(EmailProcessingResults emailProcessingResults) {
        List<SendEmailResult> failedEmails = emailProcessingResults.emailsToResend().stream()
                .map(failedEmail -> createSendEmailResult(failedEmail, false))
                .toList();

        List<SendEmailResult> successEmails = emailProcessingResults.successfullySentEmails().stream()
                .map(successfullySentEmail -> createSendEmailResult(successfullySentEmail, true))
                .toList();

        List<SendEmailResult> emailResults = new ArrayList<>();
        emailResults.addAll(failedEmails);
        emailResults.addAll(successEmails);

        return emailResults;
    }

    private SendEmailResult createSendEmailResult(SendEmail sendEmail, boolean sent) {
        return new SendEmailResult(sendEmail, sent);
    }
}
