package gov.irs.directfile.emailservice.services;

import java.util.Map;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.audit.LogUtil;
import gov.irs.directfile.emailservice.domain.SendEmail;

@Slf4j
@Service
@Profile("blackhole")
public class BlackholeSendService implements ISendService {
    private final LogUtil structuredLogger;

    public BlackholeSendService() {
        this.structuredLogger = new LogUtil(log);
    }

    @Override
    public boolean sendEmail(SendEmail email) {
        structuredLogger.logAtInfo(
                "email sent with template " + email.getEmailType().toString(),
                Map.of(
                        AuditLogElement.email,
                        email.getRecipientEmailAddress(),
                        AuditLogElement.taxReturnId,
                        email.getTaxReturnId()));
        return true;
    }
}
