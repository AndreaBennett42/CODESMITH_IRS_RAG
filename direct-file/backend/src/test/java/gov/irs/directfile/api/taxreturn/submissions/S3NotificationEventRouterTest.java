package gov.irs.directfile.api.taxreturn.submissions;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
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
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.*;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.pdfBackfill.PDFBackfillToS3Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc.PublishSubmissionConfirmationsEventHandler;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class S3NotificationEventRouterTest {

    S3NotificationEventRouter s3NotificationEventRouter;

    S3NotificationEventService s3NotificationEventService;

    S3ConfigurationProperties s3ConfigurationProperties;

    @Mock
    ConfirmationService confirmationService;

    @Mock
    ReminderEmailCacheService reminderEmailCacheService;

    @Mock
    TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Mock
    TechnicalErrorResolvedHandler technicalErrorResolvedHandler =
            new TechnicalErrorResolvedHandler(confirmationService, taxReturnSubmissionRepository);

    @Mock
    ReminderEmailHandler reminderEmailHandler;

    @Mock
    PublishDispatchMessageEventHandler publishDispatchMessageEventHandler;

    @Mock
    PublishSubmissionConfirmationsEventHandler publishSubmissionConfirmationsEventHandler;

    @Mock(name = "s3WithoutEncryption")
    S3Client mockS3Client;

    @Mock
    PdfService pdfService;

    @Mock
    TaxReturnRepository taxReturnRepository;

    @Mock
    S3StorageService s3StorageService;

    PDFBackfillToS3Handler backfillToS3Handler =
            new PDFBackfillToS3Handler(taxReturnRepository, pdfService, s3StorageService);

    String sqsMessage = "{\"path\": \"adhoc_job.json\"}";
    String technicalErrorResolvedJson =
            "{\"key\":\"technical_error_resolved\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\"]}}";
    String sendReminderEmailJson =
            "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\"]}}";
    String publishDispatchMessageJson =
            "{\"key\":\"publish_dispatch_queue_messages\",\"payload\":{\"submissionIds\":[\"1111\",\"2222\"]}}";
    String publishSubmissionConfirmationsJson =
            "{\"key\":\"publish_submission_confirmations\",\"payload\":{\"taxReturnSubmissionReceiptInformations\":[{\"submissionId\":\"1111\",\"receiptId\":\"2222\"}]}}";

    String invalidJson = "{\"key\":\"send_everyone_emails\",\"payload\":{\"ids\":[\"*\"]}}";

    @BeforeEach
    public void setup() throws JsonProcessingException {
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
    }

    @Test
    void whenS3NotificationEventServiceHandleS3NotificationEventForPostSubmissionError_callsCorrectHandler()
            throws JsonProcessingException {
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, technicalErrorResolvedJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        JsonNode payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job.json", "some-bucket")
                .get("payload");

        s3NotificationEventService.handleS3NotificationEvent(sqsMessage);

        verify(technicalErrorResolvedHandler, times(1)).handleNotificationEvent(payload);
    }

    @Test
    void whenS3NotificationEventServiceHandleS3NotificationEventForReminderEmail_callsCorrectHandler()
            throws JsonProcessingException {
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, sendReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        JsonNode payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job.json", "some-bucket")
                .get("payload");

        s3NotificationEventService.handleS3NotificationEvent(sqsMessage);

        verify(reminderEmailHandler, times(1)).handleNotificationEvent(payload);
    }

    @Test
    void
            whenS3NotificationEventServiceHandleS3NotificationEventForPublishDispatchMessageEventHandler_callsCorrectHandler()
                    throws JsonProcessingException {
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, publishDispatchMessageJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        JsonNode payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job.json", "some-bucket")
                .get("payload");

        s3NotificationEventService.handleS3NotificationEvent(sqsMessage);

        verify(publishDispatchMessageEventHandler, times(1)).handleNotificationEvent(payload);
    }

    @Test
    void
            whenS3NotificationEventServiceHandleS3NotificationEventForPublishSubmissionConfirmationsEventHandler_callsCorrectHandler()
                    throws JsonProcessingException {
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, publishSubmissionConfirmationsJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        JsonNode payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job.json", "some-bucket")
                .get("payload");

        s3NotificationEventService.handleS3NotificationEvent(sqsMessage);

        verify(publishSubmissionConfirmationsEventHandler, times(1)).handleNotificationEvent(payload);
    }

    @Test
    void whenS3NotificationEventServiceHandleS3NotificationEvent_whenHandlerTypeIsNotSupport_thenNoHandlerCalled()
            throws JsonProcessingException {
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, invalidJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        JsonNode payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job.json", "some-bucket")
                .get("payload");

        s3NotificationEventService.handleS3NotificationEvent(sqsMessage);

        verify(reminderEmailHandler, times(0)).handleNotificationEvent(payload);
    }
}
