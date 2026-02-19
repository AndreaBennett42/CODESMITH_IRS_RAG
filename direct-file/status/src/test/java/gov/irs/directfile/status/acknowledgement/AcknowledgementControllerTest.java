package gov.irs.directfile.status.acknowledgement;

import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;

import ch.qos.logback.classic.Level;
import jakarta.persistence.EntityNotFoundException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.http.HttpStatus;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.audit.AuditService;
import gov.irs.directfile.audit.events.*;
import gov.irs.directfile.models.RejectedStatus;
import gov.irs.directfile.status.acknowledgement.domain.AcknowledgementStatus;
import gov.irs.directfile.status.acknowledgement.domain.Status;
import gov.irs.directfile.status.config.SnsClientTestConfiguration;
import gov.irs.directfile.status.extension.LoggerExtension;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@TestPropertySource(properties = {"status.xml-headers-to-be-removed="})
@Import(SnsClientTestConfiguration.class)
class AcknowledgementControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private AcknowledgementService acknowledgementService;

    @RegisterExtension
    public static LoggerExtension logVerifier = new LoggerExtension(Level.INFO, AuditService.class.getName());

    private static String X_FORWARDED_FOR = "X-Forwarded-For";
    private static final String TEST_IP_ADDR1 = "10.1.2.1";
    private static final String TEST_IP_ADDR2 = "10.1.2.2";
    private static final String REMOTE_IP_ADDR = "10.1.2.3";

    @Test
    public void status_whenAcknowledgementServiceReturnsPendingStatus_endpointReturnsOkResponse() throws Exception {
        AcknowledgementStatus acknowledgementStatus =
                new AcknowledgementStatus(Status.Pending, "", List.of(), new Date());
        UUID taxReturnId = UUID.randomUUID();

        when(acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(eq(taxReturnId)))
                .thenReturn("1234567890");
        when(acknowledgementService.GetAcknowledgement(eq(taxReturnId), eq("1234567890")))
                .thenReturn(acknowledgementStatus);

        this.mockMvc
                .perform(get("/status")
                        .with(r -> {
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("id", taxReturnId.toString()))
                .andExpect(status().isOk());
        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.SUCCESS)
                        .mefSubmissionId("1234567890")
                        .build(),
                Map.of(
                        AuditLogElement.taxReturnId,
                        taxReturnId.toString(),
                        AuditLogElement.responseStatusCode,
                        HttpStatus.OK.value(),
                        AuditLogElement.remoteAddress,
                        REMOTE_IP_ADDR,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void status_whenAcknowledgementServiceReturnsNull_endpointReturnsNotFoundResponse() throws Exception {
        UUID taxReturnId = UUID.randomUUID();

        when(acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(eq(taxReturnId)))
                .thenReturn("1234567890");
        when(acknowledgementService.GetAcknowledgement(eq(taxReturnId), eq("1234567890")))
                .thenReturn(null);

        this.mockMvc
                .perform(get("/status")
                        .with(r -> {
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("id", taxReturnId.toString()))
                .andExpect(status().isNotFound());
        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.FAILURE)
                        .build(),
                Map.of(
                        AuditLogElement.taxReturnId,
                        taxReturnId.toString(),
                        AuditLogElement.responseStatusCode,
                        HttpStatus.NOT_FOUND.value(),
                        AuditLogElement.remoteAddress,
                        REMOTE_IP_ADDR,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void status_whenAcknowledgementServiceThrowsException_endpointReturnsBadRequestResponse() throws Exception {
        UUID taxReturnId = UUID.randomUUID();

        when(acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(eq(taxReturnId)))
                .thenReturn("1234567890");
        when(acknowledgementService.GetAcknowledgement(eq(taxReturnId), eq("1234567890")))
                .thenThrow(new NullPointerException("Error occurred"));

        this.mockMvc
                .perform(get("/status")
                        .with(r -> {
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("id", taxReturnId.toString()))
                .andExpect(status().isBadRequest());
        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{errorMessage=Error occurred}")
                        .eventErrorMessage("java.lang.NullPointerException")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.FAILURE)
                        .build(),
                Map.of(
                        AuditLogElement.taxReturnId,
                        taxReturnId.toString(),
                        AuditLogElement.responseStatusCode,
                        HttpStatus.BAD_REQUEST.value(),
                        AuditLogElement.remoteAddress,
                        REMOTE_IP_ADDR,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void status_endpointHitWithInvalidUUID_endpointReturnsBadRequestResponse() throws Exception {
        String invalidUUIDString = "12323123231321";

        this.mockMvc.perform(get("/status").param("id", invalidUUIDString)).andExpect(status().isBadRequest());
    }

    @Test
    public void testRemoteIpAddress_OneIpInX_Forwarded_For() throws Exception {
        AcknowledgementStatus acknowledgementStatus =
                new AcknowledgementStatus(Status.Pending, "", List.of(), new Date());
        UUID taxReturnId = UUID.randomUUID();

        when(acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(eq(taxReturnId)))
                .thenReturn("1234567890");
        when(acknowledgementService.GetAcknowledgement(eq(taxReturnId), eq("1234567890")))
                .thenReturn(acknowledgementStatus);

        // 1 ip address in x-forwarded-for
        this.mockMvc
                .perform(get("/status")
                        .with(r -> {
                            r.addHeader(X_FORWARDED_FOR, TEST_IP_ADDR1);
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("id", taxReturnId.toString()))
                .andExpect(status().isOk());
        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.SUCCESS)
                        .mefSubmissionId("1234567890")
                        .build(),
                Map.of(
                        AuditLogElement.taxReturnId,
                        taxReturnId.toString(),
                        AuditLogElement.responseStatusCode,
                        HttpStatus.OK.value(),
                        AuditLogElement.remoteAddress,
                        TEST_IP_ADDR1,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void testRemoteIpAddress_TwoIpInX_Forwarded_For() throws Exception {
        AcknowledgementStatus acknowledgementStatus =
                new AcknowledgementStatus(Status.Pending, "", List.of(), new Date());
        UUID taxReturnId = UUID.randomUUID();

        when(acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(eq(taxReturnId)))
                .thenReturn("1234567890");
        when(acknowledgementService.GetAcknowledgement(eq(taxReturnId), eq("1234567890")))
                .thenReturn(acknowledgementStatus);

        // 2 ip addresses in x-forwarded-for
        this.mockMvc
                .perform(get("/status")
                        .with(r -> {
                            r.addHeader(X_FORWARDED_FOR, TEST_IP_ADDR2 + "," + TEST_IP_ADDR1);
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("id", taxReturnId.toString()))
                .andExpect(status().isOk());
        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.SUCCESS)
                        .mefSubmissionId("1234567890")
                        .build(),
                Map.of(
                        AuditLogElement.taxReturnId,
                        taxReturnId.toString(),
                        AuditLogElement.responseStatusCode,
                        HttpStatus.OK.value(),
                        AuditLogElement.remoteAddress,
                        TEST_IP_ADDR2,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void rejectionCodes_whenAcknowledgementServiceReturnsRejectionCodes_endpointReturnsOkResponse()
            throws Exception {
        List<RejectedStatus> rejectionCodes = List.of(
                new RejectedStatus("code1", "key1", "description1"),
                new RejectedStatus("code2", "key2", "description2"));

        String submissionId = UUID.randomUUID().toString();

        when(acknowledgementService.getRejectionCodesForSubmissionId(submissionId))
                .thenReturn(rejectionCodes);

        this.mockMvc
                .perform(get("/status/rejection-codes")
                        .with(r -> {
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("submissionId", submissionId))
                .andExpect(status().isOk());

        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.SUCCESS)
                        .mefSubmissionId(submissionId)
                        .build(),
                Map.of(
                        AuditLogElement.mefSubmissionId,
                        submissionId,
                        AuditLogElement.responseStatusCode,
                        HttpStatus.OK.value(),
                        AuditLogElement.remoteAddress,
                        REMOTE_IP_ADDR,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void rejectionCodes_whenAcknowledgementServiceThrowsNotFoundException_endpointReturnsNotFoundResponse()
            throws Exception {
        String submissionId = UUID.randomUUID().toString();

        when(acknowledgementService.getRejectionCodesForSubmissionId(submissionId))
                .thenThrow(EntityNotFoundException.class);

        this.mockMvc
                .perform(get("/status/rejection-codes")
                        .with(r -> {
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("submissionId", submissionId))
                .andExpect(status().isNotFound());

        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.FAILURE)
                        .mefSubmissionId(submissionId)
                        .build(),
                Map.of(
                        AuditLogElement.mefSubmissionId,
                        submissionId,
                        AuditLogElement.responseStatusCode,
                        HttpStatus.NOT_FOUND.value(),
                        AuditLogElement.remoteAddress,
                        REMOTE_IP_ADDR,
                        AuditLogElement.cyberOnly,
                        true));
    }

    @Test
    public void rejectionCodes_whenAcknowledgementServiceThrowsException_endpointReturnsBadRequestResponse()
            throws Exception {
        String submissionId = UUID.randomUUID().toString();

        when(acknowledgementService.getRejectionCodesForSubmissionId(submissionId))
                .thenThrow(new NullPointerException("Error occurred"));

        this.mockMvc
                .perform(get("/status/rejection-codes")
                        .with(r -> {
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("submissionId", submissionId))
                .andExpect(status().isBadRequest());

        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{errorMessage=Error occurred}")
                        .eventErrorMessage("java.lang.NullPointerException")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.FAILURE)
                        .mefSubmissionId(submissionId)
                        .build(),
                Map.of(
                        AuditLogElement.mefSubmissionId,
                        submissionId,
                        AuditLogElement.responseStatusCode,
                        HttpStatus.BAD_REQUEST.value(),
                        AuditLogElement.remoteAddress,
                        REMOTE_IP_ADDR,
                        AuditLogElement.cyberOnly,
                        true));
    }

    private static Stream<Arguments> getRejectionCodesXForwardedForParameters() {
        return Stream.of(
                Arguments.of(TEST_IP_ADDR1, TEST_IP_ADDR1),
                Arguments.of(TEST_IP_ADDR2 + "," + TEST_IP_ADDR1, TEST_IP_ADDR2));
    }

    @ParameterizedTest
    @MethodSource("getRejectionCodesXForwardedForParameters")
    public void rejectionCodes_whenXForwardedFor_endpointReturnsOkResponse(
            String xForwardedForValue, String expectedRemoteAddress) throws Exception {
        List<RejectedStatus> rejectionCodes = List.of(
                new RejectedStatus("code1", "key1", "description1"),
                new RejectedStatus("code2", "key2", "description2"));

        String submissionId = UUID.randomUUID().toString();

        when(acknowledgementService.getRejectionCodesForSubmissionId(submissionId))
                .thenReturn(rejectionCodes);

        this.mockMvc
                .perform(get("/status/rejection-codes")
                        .with(r -> {
                            r.addHeader(X_FORWARDED_FOR, xForwardedForValue);
                            r.setRemoteAddr(REMOTE_IP_ADDR);
                            return r;
                        })
                        .param("submissionId", submissionId))
                .andExpect(status().isOk());

        // verify audit log message
        logVerifier.verifyLogEvent(
                Event.builder()
                        .detail("{}")
                        .eventId(EventId.CHECK)
                        .eventPrincipal(new SystemEventPrincipal())
                        .eventStatus(EventStatus.SUCCESS)
                        .mefSubmissionId(submissionId)
                        .build(),
                Map.of(
                        AuditLogElement.mefSubmissionId,
                        submissionId,
                        AuditLogElement.responseStatusCode,
                        HttpStatus.OK.value(),
                        AuditLogElement.remoteAddress,
                        expectedRemoteAddress,
                        AuditLogElement.cyberOnly,
                        true));
    }
}
