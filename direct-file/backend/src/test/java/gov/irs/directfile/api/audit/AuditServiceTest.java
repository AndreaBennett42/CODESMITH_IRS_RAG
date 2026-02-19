package gov.irs.directfile.api.audit;

import ch.qos.logback.classic.Level;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.extension.LoggerExtension;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class AuditServiceTest {
    @InjectMocks
    private AuditService auditService;

    @Mock
    private AuditEventContextHolder auditEventContextHolder;

    @RegisterExtension
    public static LoggerExtension logVerifier = new LoggerExtension(Level.INFO, AuditService.class.getName());

    @SneakyThrows
    @Test
    public void givenValidLogElementName_whenAddEventPropertiesForUserTin_thenUserTinLogAdded() {
        // given
        // when
        auditService.addEventProperty(AuditLogElement.USER_TIN, "test-tin");

        // then
        verify(auditEventContextHolder, times(1)).addValueToEventMap(AuditLogElement.USER_TIN, "test-tin");
    }

    @SneakyThrows
    @Test
    public void givenValidLogElementName_whenAddEventPropertiesForTaxYear_thenTaxYearLogAdded() {
        // given
        // when
        auditService.addEventProperty(AuditLogElement.TAX_PERIOD, "2022");

        // then
        verify(auditEventContextHolder, times(1)).addValueToEventMap(AuditLogElement.TAX_PERIOD, "2022");
    }
}
