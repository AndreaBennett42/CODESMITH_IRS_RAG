package gov.irs.directfile.submit.service;

import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.MockedConstruction;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;

import gov.irs.a2a.mef.mefheader.ErrorClassificationCdType;
import gov.irs.a2a.mef.mefheader.ErrorExceptionDetailType;
import gov.irs.a2a.mef.meftransmitterservicemtom.ErrorExceptionDetail;
import gov.irs.mef.exception.ServiceException;
import gov.irs.mef.exception.ToolkitException;
import gov.irs.mef.inputcomposition.SubmissionContainer;
import gov.irs.mef.services.transmitter.mtom.SendSubmissionsMTOMClient;
import gov.irs.mef.services.transmitter.mtom.SendSubmissionsResult;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.audit.AuditService;
import gov.irs.directfile.audit.events.Event;
import gov.irs.directfile.audit.events.EventId;
import gov.irs.directfile.audit.events.EventStatus;
import gov.irs.directfile.audit.events.SystemEventPrincipal;
import gov.irs.directfile.audit.events.TinType;
import gov.irs.directfile.submit.actions.ActionContext;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.SnsClientTestConfiguration;
import gov.irs.directfile.submit.config.SynchronousS3TestConfiguration;
import gov.irs.directfile.submit.domain.BundledArchives;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.UserContextData;
import gov.irs.directfile.submit.extension.LoggerExtension;

import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@SpringBootTest
@Import({SynchronousS3TestConfiguration.class, SnsClientTestConfiguration.class})
public class MefBundleSubmissionServiceTest {
    @Autowired
    Config config;

    @MockBean
    SqsConnectionSetupService sqsConnectionSetupService;

    @RegisterExtension
    public static LoggerExtension loggerExtension = new LoggerExtension(Level.INFO, AuditService.class.getName());

    final UserContextData userContextData = new UserContextData(
            "00000",
            "00000000-0000-0000-0000-000000000000",
            "11111111-1111-1111-1111-111111111111",
            "111001111",
            TinType.INDIVIDUAL,
            "0.0.0.0",
            "2024-01-01");

    @Test
    void mefBundleSubmissionServiceLogsSubmitBatchSuccessAuditEvent() throws Exception {
        // given
        Event expectedEvent = Event.builder()
                .eventId(EventId.SUBMIT_BATCH)
                .eventStatus(EventStatus.SUCCESS)
                .eventPrincipal(new SystemEventPrincipal())
                .detail("{mefSubmissionIds=00000}")
                .build();

        final SubmissionContainer submissionContainer = mock(SubmissionContainer.class);
        final SendSubmissionsResult sendSubmissionsResult = mock(SendSubmissionsResult.class);

        BundledArchives bundledArchives =
                new BundledArchives((List<UserContextData>) List.of(userContextData), submissionContainer);
        MefBundleSubmissionActionHandler mefBundleSubmissionService =
                new MefBundleSubmissionActionHandler(config, new ActionContext(config));

        try (MockedConstruction<SendSubmissionsMTOMClient> mockSendSubmissionsMTOClient =
                Mockito.mockConstruction(SendSubmissionsMTOMClient.class, (sendSubmissionsMTOClient, context) -> {
                    when(sendSubmissionsMTOClient.invoke(any(), any())).thenReturn(sendSubmissionsResult);
                })) {

            // when
            mefBundleSubmissionService.submitBundles(bundledArchives, new SubmissionBatch(0L, ""));

            // then verify 1 event was logged with expected values
            loggerExtension.verifyLogEvent(
                    expectedEvent,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "200",
                            AuditLogElement.remoteAddress,
                            userContextData.getRemoteAddress()));
        }
    }

    @Test
    void submitBundleActionLogsSubmitBatchFailureAuditEvent() {
        // given
        final SubmissionContainer submissionContainer = mock(SubmissionContainer.class);

        BundledArchives bundledArchives =
                new BundledArchives((List<UserContextData>) List.of(userContextData), submissionContainer);
        MefBundleSubmissionActionHandler mefBundleSubmissionService =
                new MefBundleSubmissionActionHandler(config, new ActionContext(config));

        // with ServiceException
        try (MockedConstruction<SendSubmissionsMTOMClient> mockSendSubmissionsMTOClient =
                Mockito.mockConstruction(SendSubmissionsMTOMClient.class, (sendSubmissionsMTOClient, context) -> {
                    ErrorExceptionDetailType errorExceptionDetailType = new ErrorExceptionDetailType();
                    errorExceptionDetailType.setErrorClassificationCd(ErrorClassificationCdType.SYSTEM_ERROR);
                    final ServiceException serviceException = new ServiceException(
                            "test service exception",
                            new ErrorExceptionDetail("test error exception detail", errorExceptionDetailType),
                            java.util.logging.Level.SEVERE);

                    when(sendSubmissionsMTOClient.invoke(any(), any())).thenThrow(serviceException);
                })) {

            // when
            mefBundleSubmissionService.submitBundles(bundledArchives, new SubmissionBatch(0L, ""));
            fail("expected it to throw exception");
        } catch (Exception exception) {
            Event expectedEvent = Event.builder()
                    .eventId(EventId.SUBMIT_BATCH)
                    .eventStatus(EventStatus.FAILURE)
                    .eventPrincipal(new SystemEventPrincipal())
                    .eventErrorMessage("java.lang.RuntimeException")
                    .detail(
                            "{mefSubmissionIds=00000, errorMessage=gov.irs.mef.exception.ServiceException: test service exception}")
                    .build();
            loggerExtension.verifyLogEvent(
                    expectedEvent,
                    0,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "400",
                            AuditLogElement.eventErrorMessage,
                            exception.getCause().getClass().getName(),
                            AuditLogElement.remoteAddress,
                            userContextData.getRemoteAddress()));
        }

        // with ToolkitException
        try (MockedConstruction<SendSubmissionsMTOMClient> mockSendSubmissionsMTOClient =
                Mockito.mockConstruction(SendSubmissionsMTOMClient.class, (sendSubmissionsMTOClient, context) -> {
                    when(sendSubmissionsMTOClient.invoke(any(), any()))
                            .thenThrow(new ToolkitException("test toolkit exception"));
                })) {

            // when
            mefBundleSubmissionService.submitBundles(bundledArchives, new SubmissionBatch(0L, ""));
            fail("expected it to throw exception");
        } catch (Exception exception) {
            Event expectedEvent = Event.builder()
                    .eventId(EventId.SUBMIT_BATCH)
                    .eventStatus(EventStatus.FAILURE)
                    .eventPrincipal(new SystemEventPrincipal())
                    .eventErrorMessage("java.lang.RuntimeException")
                    .detail(
                            "{mefSubmissionIds=00000, errorMessage=gov.irs.mef.exception.ToolkitException: test toolkit exception}")
                    .build();
            loggerExtension.verifyLogEvent(
                    expectedEvent,
                    1,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "400",
                            AuditLogElement.eventErrorMessage,
                            exception.getCause().getClass().getName(),
                            AuditLogElement.remoteAddress,
                            userContextData.getRemoteAddress()));
        }
    }
}
