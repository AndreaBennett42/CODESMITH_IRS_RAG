package gov.irs.directfile.emailservice.services;

import jakarta.mail.MessagingException;

import gov.irs.directfile.emailservice.domain.SendEmail;

public interface ISendService {
    boolean sendEmail(SendEmail email) throws MessagingException;
}
