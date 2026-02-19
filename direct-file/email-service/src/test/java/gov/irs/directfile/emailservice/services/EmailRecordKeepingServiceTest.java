package gov.irs.directfile.emailservice.services;

import java.util.List;
import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.emailservice.domain.SendEmail;
import gov.irs.directfile.emailservice.repositories.SendEmailRepository;
import gov.irs.directfile.emailservice.repositories.SendEmailResultRepository;
import gov.irs.directfile.models.email.HtmlTemplate;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class EmailRecordKeepingServiceTest {
    EmailRecordKeepingService emailRecordKeepingService;

    @Mock
    SendEmailRepository sendEmailRepository;

    @Mock
    SendEmailResultRepository sendEmailResultRepository;

    @BeforeEach
    public void setup() {
        emailRecordKeepingService = new EmailRecordKeepingService(sendEmailRepository, sendEmailResultRepository);
    }

    @Test
    public void recordSendEmailAttempts_givenSendEmails_persistsSendEmailRecords() {
        SendEmail sendEmail1 = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "person1@aol.com",
                null,
                "en",
                HtmlTemplate.SUBMITTED);
        SendEmail sendEmail2 = new SendEmail(
                UUID.randomUUID(),
                "submissionId",
                UUID.randomUUID(),
                "person2@aol.com",
                null,
                "en",
                HtmlTemplate.SUBMITTED);
        List<SendEmail> sendEmails = List.of(sendEmail1, sendEmail2);

        emailRecordKeepingService.recordSendEmails(sendEmails);

        verify(sendEmailRepository, times(1)).saveAll(sendEmails);
    }
}
