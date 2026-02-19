package gov.irs.directfile.submit.actions;

import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;

import gov.irs.mef.exception.ToolkitException;
import gov.irs.mef.inputcomposition.PostmarkedSubmissionArchive;
import gov.irs.mef.inputcomposition.SubmissionBuilder;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.audit.AuditService;
import gov.irs.directfile.audit.events.Event;
import gov.irs.directfile.audit.events.EventId;
import gov.irs.directfile.audit.events.EventStatus;
import gov.irs.directfile.audit.events.SystemEventPrincipal;
import gov.irs.directfile.audit.events.TaxPeriod;
import gov.irs.directfile.audit.events.TinType;
import gov.irs.directfile.audit.events.XXXCode;
import gov.irs.directfile.submit.actions.results.CreateArchiveActionResult;
import gov.irs.directfile.submit.command.BundleArchiveAction;
import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.SnsClientTestConfiguration;
import gov.irs.directfile.submit.config.SynchronousS3TestConfiguration;
import gov.irs.directfile.submit.domain.SubmissionArchiveContainer;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.UserContextData;
import gov.irs.directfile.submit.extension.LoggerExtension;

import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;

@SpringBootTest
@Import({SynchronousS3TestConfiguration.class, SnsClientTestConfiguration.class})
public class BundleArchivesActionHandlerTest {

    // NOTE: Needed because we're mocking MEF Classes. MeF SDK expects this env variable A2A_TOOLKIT_HOME to be defined
    @BeforeAll
    public static void setupSystemProperties() {
        String userDirectory = System.getProperty("user.dir");
        System.setProperty("A2A_TOOLKIT_HOME", userDirectory + "/src/test/resources/");
    }

    @AfterAll
    public static void cleanupSystemProperties() {
        System.clearProperty("A2A_TOOLKIT_HOME");
    }

    @Autowired
    Config config;

    @RegisterExtension
    public static LoggerExtension logVerifier = new LoggerExtension(Level.INFO, AuditService.class.getName());

    // default test data
    final UserContextData userContextData1 = new UserContextData(
            "00000",
            "00000000-0000-0000-0000-000000000000",
            "11111111-1111-1111-1111-111111111111",
            "111001111",
            TinType.INDIVIDUAL,
            "0.0.0.0",
            "2024-01-01");
    final UserContextData userContextData2 = new UserContextData(
            "11111",
            "88888888-8888-8888-8888-888888888888",
            "99999999-9999-9999-9999-999999999999",
            "111002222",
            TinType.INDIVIDUAL,
            "1.1.1.1",
            "2024-01-01");
    final PostmarkedSubmissionArchive mockSubmissionArchive1 = mock(PostmarkedSubmissionArchive.class);
    final PostmarkedSubmissionArchive mockSubmissionArchive2 = mock(PostmarkedSubmissionArchive.class);

    final List<SubmissionArchiveContainer> submissionArchiveContainers = List.of(
            new SubmissionArchiveContainer(userContextData1, mockSubmissionArchive1),
            new SubmissionArchiveContainer(userContextData2, mockSubmissionArchive2));

    @Test
    void bundleArchivesActionLogsCreateBundleSuccessAuditEvent() throws ActionException {
        // given
        ActionContext actionContext = new ActionContext(config);
        SubmissionBatch submissionBatch = new SubmissionBatch(0L, "");
        CreateArchiveActionResult createArchiveActionResult =
                new CreateArchiveActionResult(submissionBatch, submissionArchiveContainers);
        BundleArchivesActionHandler bundleArchivesActionHandler = new BundleArchivesActionHandler(actionContext);

        try (MockedStatic<SubmissionBuilder> mockSubmissionBuilder = Mockito.mockStatic(SubmissionBuilder.class)) {
            mockSubmissionBuilder
                    .when(() -> SubmissionBuilder.createPostmarkedSubmissionArchive(any(), any()))
                    .thenReturn(mockSubmissionArchive1)
                    .thenReturn(mockSubmissionArchive2);

            // when
            bundleArchivesActionHandler.handleBundleCommand(new BundleArchiveAction(createArchiveActionResult));

            // then verify 2 events were logged with expected values
            logVerifier.verifyLogEvent(
                    generateCreateBundleSuccessAuditEvent("{directFileUserId=00000000-0000-0000-0000-000000000000}"),
                    0,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "200",
                            AuditLogElement.mefSubmissionId,
                            userContextData1.getSubmissionId(),
                            AuditLogElement.taxPeriod,
                            TaxPeriod.TY2024,
                            AuditLogElement.taxReturnId,
                            userContextData1.getTaxReturnId(),
                            AuditLogElement.userTin,
                            userContextData1.getUserTin(),
                            AuditLogElement.userTinType,
                            userContextData1.getUserTinType(),
                            AuditLogElement.mftCode,
                            XXXCode.XXX_CODE,
                            AuditLogElement.remoteAddress,
                            userContextData1.getRemoteAddress()));

            logVerifier.verifyLogEvent(
                    generateCreateBundleSuccessAuditEvent("{directFileUserId=88888888-8888-8888-8888-888888888888}"),
                    1,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "200",
                            AuditLogElement.mefSubmissionId,
                            userContextData2.getSubmissionId(),
                            AuditLogElement.taxPeriod,
                            TaxPeriod.TY2024,
                            AuditLogElement.taxReturnId,
                            userContextData2.getTaxReturnId(),
                            AuditLogElement.userTin,
                            userContextData2.getUserTin(),
                            AuditLogElement.userTinType,
                            userContextData2.getUserTinType(),
                            AuditLogElement.mftCode,
                            XXXCode.XXX_CODE,
                            AuditLogElement.remoteAddress,
                            userContextData2.getRemoteAddress()));
        }
    }

    @Test
    void bundleArchivesActionLogsCreateBundleFailureAuditEvent() {
        // given
        ActionContext actionContext = new ActionContext(config);
        SubmissionBatch submissionBatch = new SubmissionBatch(0L, "");
        CreateArchiveActionResult createArchiveActionResult =
                new CreateArchiveActionResult(submissionBatch, submissionArchiveContainers);
        BundleArchivesActionHandler bundleArchivesActionHandler = new BundleArchivesActionHandler(actionContext);

        try (MockedStatic<SubmissionBuilder> mockSubmissionBuilder = Mockito.mockStatic(SubmissionBuilder.class)) {
            mockSubmissionBuilder
                    .when(() -> SubmissionBuilder.createSubmissionContainer(
                            any(PostmarkedSubmissionArchive[].class), anyString()))
                    .thenThrow(new ToolkitException("test toolkit exception"));

            // when
            bundleArchivesActionHandler.handleBundleCommand(new BundleArchiveAction(createArchiveActionResult));
            fail("expected it to throw an exception");
        } catch (ActionException actionException) {
            // then verify 2 events were logged with expected values
            logVerifier.verifyLogEvent(
                    generateCreateBundleFailureAuditEvent(
                            "gov.irs.mef.exception.ToolkitException",
                            "{directFileUserId=00000000-0000-0000-0000-000000000000, errorMessage=test toolkit exception}"),
                    0,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "400",
                            AuditLogElement.mefSubmissionId,
                            userContextData1.getSubmissionId(),
                            AuditLogElement.taxPeriod,
                            TaxPeriod.TY2024,
                            AuditLogElement.taxReturnId,
                            userContextData1.getTaxReturnId(),
                            AuditLogElement.mftCode,
                            XXXCode.XXX_CODE,
                            AuditLogElement.remoteAddress,
                            userContextData1.getRemoteAddress()));

            logVerifier.verifyLogEvent(
                    generateCreateBundleFailureAuditEvent(
                            "gov.irs.mef.exception.ToolkitException",
                            "{directFileUserId=88888888-8888-8888-8888-888888888888, errorMessage=test toolkit exception}"),
                    1,
                    Map.of(
                            AuditLogElement.cyberOnly,
                            true,
                            AuditLogElement.responseStatusCode,
                            "400",
                            AuditLogElement.mefSubmissionId,
                            userContextData2.getSubmissionId(),
                            AuditLogElement.taxPeriod,
                            TaxPeriod.TY2024,
                            AuditLogElement.taxReturnId,
                            userContextData2.getTaxReturnId(),
                            AuditLogElement.mftCode,
                            XXXCode.XXX_CODE,
                            AuditLogElement.remoteAddress,
                            userContextData2.getRemoteAddress()));
        }
    }

    private Event generateCreateBundleSuccessAuditEvent(String detail) {
        return Event.builder()
                .eventId(EventId.CREATE_BUNDLE)
                .eventStatus(EventStatus.SUCCESS)
                .eventPrincipal(new SystemEventPrincipal())
                .detail(detail)
                .build();
    }

    private Event generateCreateBundleFailureAuditEvent(String eventErrorMessage, String detail) {
        return Event.builder()
                .eventId(EventId.CREATE_BUNDLE)
                .eventStatus(EventStatus.FAILURE)
                .eventErrorMessage(eventErrorMessage)
                .eventPrincipal(new SystemEventPrincipal())
                .detail(detail)
                .build();
    }
}
