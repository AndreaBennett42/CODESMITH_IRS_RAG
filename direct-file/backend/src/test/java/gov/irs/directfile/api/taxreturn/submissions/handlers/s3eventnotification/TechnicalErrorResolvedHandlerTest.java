package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.util.Optional;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;

import gov.irs.directfile.api.config.S3ConfigurationProperties;
import gov.irs.directfile.api.config.S3ConfigurationProperties.S3;
import gov.irs.directfile.api.io.documentstore.S3StorageService;
import gov.irs.directfile.api.pdf.PdfService;
import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.api.taxreturn.submissions.S3NotificationEventRouter;
import gov.irs.directfile.api.taxreturn.submissions.S3NotificationEventService;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.pdfBackfill.PDFBackfillToS3Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc.PublishSubmissionConfirmationsEventHandler;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class TechnicalErrorResolvedHandlerTest {
    S3NotificationEventRouter s3NotificationEventRouter;

    S3NotificationEventService s3NotificationEventService;

    S3ConfigurationProperties s3ConfigurationProperties;

    @Mock
    ConfirmationService confirmationService;

    @Mock
    TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Mock
    ReminderEmailHandler reminderEmailHandler;

    @Mock
    PublishDispatchMessageEventHandler publishDispatchMessageEventHandler;

    @Mock
    PublishSubmissionConfirmationsEventHandler publishSubmissionConfirmationsEventHandler;

    @Mock(name = "s3WithoutEncryption")
    S3Client mockS3Client;

    String sqsMessage = "{\"path\": \"adhoc_job.json\"}";
    String technicalErrorResolvedJson =
            "{\"key\":\"technical_error_resolved\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\",\"ce019609-99e0-4ef5-85bb-ad90dc302e71\",\"ce019609-99e0-4ef5-85bb-ad90dc302e72\"]}}";
    JsonNode payload;

    @Mock
    PdfService pdfService;

    @Mock
    TaxReturnRepository taxReturnRepository;

    @Mock
    S3StorageService s3StorageService;

    PDFBackfillToS3Handler backfillToS3Handler =
            new PDFBackfillToS3Handler(taxReturnRepository, pdfService, s3StorageService);

    @BeforeEach
    public void setup() {
        TechnicalErrorResolvedHandler technicalErrorResolvedHandler =
                new TechnicalErrorResolvedHandler(confirmationService, taxReturnSubmissionRepository);
        s3NotificationEventRouter = new S3NotificationEventRouter(
                technicalErrorResolvedHandler,
                reminderEmailHandler,
                backfillToS3Handler,
                publishDispatchMessageEventHandler,
                publishSubmissionConfirmationsEventHandler);
        s3ConfigurationProperties = new S3ConfigurationProperties(
                null, null, new S3("", "", 0, "", "", "some-bucket", "some-operations-jobs-bucket", "dev"));
        s3NotificationEventService =
                new S3NotificationEventService(s3NotificationEventRouter, mockS3Client, s3ConfigurationProperties);
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, technicalErrorResolvedJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job.json", "some-bucket")
                .get("payload");
    }

    @Test
    @SneakyThrows
    void whenHandleNotificationEvent_callsEnqueueErrorResolutionEmail() {
        TechnicalErrorResolvedHandler technicalErrorResolvedHandler =
                new TechnicalErrorResolvedHandler(confirmationService, taxReturnSubmissionRepository);
        TaxReturn tr = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs = tr.addTaxReturnSubmission();
        trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
        trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
        trs.addSubmissionEvent(SubmissionEventTypeEnum.POST_SUBMISSION_ERROR);
        doReturn(Optional.of(trs))
                .when(taxReturnSubmissionRepository)
                .findLatestTaxReturnSubmissionByTaxReturnId(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        technicalErrorResolvedHandler.handleNotificationEvent(payload);

        verify(confirmationService, times(1)).enqueueErrorResolutionEmail(any());
        verify(taxReturnSubmissionRepository, times(3)).findLatestTaxReturnSubmissionByTaxReturnId(any());
    }
}
