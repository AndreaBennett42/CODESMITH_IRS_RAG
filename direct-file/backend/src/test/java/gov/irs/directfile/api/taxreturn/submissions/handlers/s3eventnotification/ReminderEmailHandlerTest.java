package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.ScrollPosition;
import org.springframework.data.domain.Window;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;

import gov.irs.directfile.api.config.S3ConfigurationProperties;
import gov.irs.directfile.api.config.S3ConfigurationProperties.S3;
import gov.irs.directfile.api.io.documentstore.S3StorageService;
import gov.irs.directfile.api.pdf.PdfService;
import gov.irs.directfile.api.taxreturn.SimpleTaxReturnProjection;
import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.dto.SimpleTaxReturnImpl;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.taxreturn.submissions.*;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.pdfBackfill.PDFBackfillToS3Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc.PublishSubmissionConfirmationsEventHandler;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ReminderEmailHandlerTest {
    S3NotificationEventRouter s3NotificationEventRouter;

    S3NotificationEventService s3NotificationEventService;

    S3ConfigurationProperties s3ConfigurationProperties;

    ReminderEmailCacheService reminderEmailCacheService;

    @Mock
    ConfirmationService confirmationService;

    @Mock
    TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Mock
    TechnicalErrorResolvedHandler technicalErrorResolvedHandler;

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

    JsonNode payload;

    int BATCH_SIZE = 50;
    int MAX_CACHE_SIZE = 50000;

    @BeforeEach
    public void setup() throws JsonProcessingException {
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
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
    @SneakyThrows
    void whenHandleNotificationEvent_whenMAReminderEmail_callsEnqueueStatusChangeEmailCorrectly() {
        String maReminderEmailJson =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\",\"de019609-99e0-4ef5-85bb-ad90dc302e70\",\"ee019609-99e0-4ef5-85bb-ad90dc302e70\"],\"reminder_category_key\":\"ma\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, maReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_ma_reminder_email.json", "some-bucket")
                .get("payload");
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
        List<TaxReturnSubmission> submissionList = new ArrayList<>();
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i <= 3; i++) {
            TaxReturn tr = TaxReturn.testObjectFactory();
            TaxReturnSubmission trs = tr.addTaxReturnSubmission();
            trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
            trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
            submissionList.add(trs);
            idList.add(tr.getId());
        }
        doReturn(submissionList).when(taxReturnSubmissionRepository).findLatestTaxReturnSubmissions(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        // process payload and put items into cache
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("ma"), 3);

        // fetch items from cache and enqueue status change emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();

        verify(confirmationService, times(1)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_STATE));
        verify(taxReturnSubmissionRepository, times(1)).findLatestTaxReturnSubmissions(any());
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
    }

    @Test
    @SneakyThrows
    void whenHandleNotificationEvent_whenNYReminderEmail_callsEnqueueStatusChangeEmailCorrectly() {
        String nyReminderEmailJson =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\",\"de019609-99e0-4ef5-85bb-ad90dc302e70\",\"ee019609-99e0-4ef5-85bb-ad90dc302e70\"],\"reminder_category_key\":\"ny\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, nyReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_ny_reminder_email.json", "some-bucket")
                .get("payload");
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
        List<TaxReturnSubmission> submissionList = new ArrayList<>();
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i <= 3; i++) {
            TaxReturn tr = TaxReturn.testObjectFactory();
            TaxReturnSubmission trs = tr.addTaxReturnSubmission();
            trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
            trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
            submissionList.add(trs);
            idList.add(tr.getId());
        }
        doReturn(submissionList).when(taxReturnSubmissionRepository).findLatestTaxReturnSubmissions(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        // process payload and put items into cache
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("ny"), 3);

        // fetch items from cache and enqueue status change emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();

        verify(confirmationService, times(1)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_STATE));
        verify(taxReturnSubmissionRepository, times(1)).findLatestTaxReturnSubmissions(any());
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
    }

    @Test
    @SneakyThrows
    void whenHandleNotificationEvent_whenAZReminderEmail_callsEnqueueStatusChangeEmailCorrectly() {
        String azReminderEmailJson =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\",\"de019609-99e0-4ef5-85bb-ad90dc302e70\",\"ee019609-99e0-4ef5-85bb-ad90dc302e70\"],\"reminder_category_key\":\"az\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, azReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_ny_reminder_email.json", "some-bucket")
                .get("payload");
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
        List<TaxReturnSubmission> submissionList = new ArrayList<>();
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i <= 3; i++) {
            TaxReturn tr = TaxReturn.testObjectFactory();
            TaxReturnSubmission trs = tr.addTaxReturnSubmission();
            trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
            trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
            submissionList.add(trs);
            idList.add(tr.getId());
        }
        doReturn(submissionList).when(taxReturnSubmissionRepository).findLatestTaxReturnSubmissions(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        // process payload and put items into cache
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("az"), 3);

        // fetch items from cache and enqueue status change emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();

        verify(confirmationService, times(1)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_STATE));
        verify(taxReturnSubmissionRepository, times(1)).findLatestTaxReturnSubmissions(any());
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
    }

    @Test
    @SneakyThrows
    void whenHandleNotificationEvent_whenSubmitReminderEmail_callsEnqueueStatusChangeEmailCorrectly() {
        String submitReminderEmailJson =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\",\"de019609-99e0-4ef5-85bb-ad90dc302e70\",\"ee019609-99e0-4ef5-85bb-ad90dc302e70\"],\"reminder_category_key\":\"submit\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, submitReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_ny_reminder_email.json", "some-bucket")
                .get("payload");
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
        List<TaxReturnSubmission> submissionList = new ArrayList<>();
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i <= 3; i++) {
            TaxReturn tr = TaxReturn.testObjectFactory();
            TaxReturnSubmission trs = tr.addTaxReturnSubmission();
            trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
            trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
            submissionList.add(trs);
            idList.add(tr.getId());
        }
        doReturn(submissionList).when(taxReturnSubmissionRepository).findLatestTaxReturnSubmissions(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        // process payload and put items into cache
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("submit"), 3);

        // fetch items from cache and enqueue status change emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();

        verify(confirmationService, times(1)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_SUBMIT));
        verify(taxReturnSubmissionRepository, times(1)).findLatestTaxReturnSubmissions(any());
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
    }

    @Test
    @SneakyThrows
    void whenHandleNotificationEvent_whenResubmitReminderEmail_callsEnqueueStatusChangeEmailCorrectly() {
        String resubmitReminderEmailJson =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"ce019609-99e0-4ef5-85bb-ad90dc302e70\",\"de019609-99e0-4ef5-85bb-ad90dc302e70\",\"ee019609-99e0-4ef5-85bb-ad90dc302e70\"],\"reminder_category_key\":\"resubmit\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, resubmitReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_ny_reminder_email.json", "some-bucket")
                .get("payload");
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
        List<TaxReturnSubmission> submissionList = new ArrayList<>();
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i <= 3; i++) {
            TaxReturn tr = TaxReturn.testObjectFactory();
            TaxReturnSubmission trs = tr.addTaxReturnSubmission();
            trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
            trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
            submissionList.add(trs);
            idList.add(tr.getId());
        }
        doReturn(submissionList).when(taxReturnSubmissionRepository).findLatestTaxReturnSubmissions(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        // process payload and put items into cache
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("resubmit"), 3);

        // fetch items from cache and enqueue status change emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();

        verify(confirmationService, times(1)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_RESUBMIT));
        verify(taxReturnSubmissionRepository, times(1)).findLatestTaxReturnSubmissions(any());
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
    }

    @Test
    @SneakyThrows
    void
            whenHandleNotificationEvent_whenCacheSizeIsGreaterThanBatchSize_thenTaxReturnIdsAreStillInCacheAfterSingleProcessing() {
        String resubmitReminderEmailJson =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[\"b7bca785-92bd-4104-b9b2-cd1db99533d2\",\"56d1f624-ed59-4f32-95e2-58c826a48a54\",\"850cbdec-6fbd-4526-b7d1-962f8c8bbdf3\",\"c9fcfda7-bbae-4556-9f12-8d85d032daf6\",\"f9241252-ccb1-4a8d-ac26-dfaf8f697450\",\"8d12c7a0-e814-4ccb-b883-94f09bed5e70\",\"f59cc6fd-41f4-4913-9b7c-1dc1a4374746\",\"b01c7420-8f71-4960-a36a-749ce8c91009\",\"97610fdb-f952-437e-be60-e91417669810\",\"c73467a6-60d0-42a7-8e10-70fa065cbbf2\",\"cbdeb2b7-4305-4155-a9aa-da4cd6940743\",\"14f82779-b52e-47be-b3f9-7b5356824111\",\"99642e17-3d11-464c-8c63-c863364b2080\",\"9819fae7-01bb-4b27-b834-2ea37bac28fd\",\"4c3b5b53-46f2-4f2a-9eab-97a2b91497ac\",\"5e35c738-19de-458e-a2bd-2e598c912500\",\"18808c49-73b7-4001-b5c9-a796b041f493\",\"983691cd-8e4d-4ea1-a92c-efbbcd940623\",\"50524c00-f831-4eb5-913d-1ad6971b60b3\",\"63baaefe-db7c-4fed-b1f9-fc17a32d64d3\",\"b5b86e90-4546-4d65-a6e2-4fbe6ae88ab0\",\"4db28832-f37b-4948-8dad-a4c5cc7b202a\",\"4ee5dfc0-ff09-4131-a956-48db843fa266\",\"f89995a5-b3bd-4908-881b-a31c5695de78\",\"c756b2c9-ac16-43d0-a121-fda1a70978f2\",\"f2367141-7706-4335-b8d4-19db98a521e0\",\"84ce399d-ec22-4765-a80e-0ab6af4fc365\",\"913dd781-7dc1-4840-b18c-84e189dc5bbc\",\"e210c109-478e-463a-b06e-b9a41a3d30af\",\"19e65b08-d867-4283-b514-3b952601b697\",\"f9d2f1ea-1018-4070-b888-45eff2d6d4b4\",\"9c758f85-12a4-4693-90ec-b9498afaa8e1\",\"2bacb918-0066-402e-a084-857f4dd688fb\",\"2a7d355a-fe19-42ef-af07-90a6fe54a34d\",\"03bce671-19a3-40c9-9312-d4adadbc0d0c\",\"72198751-0b77-4b0a-94c1-71b2ad1ee9ff\",\"8caa3377-1408-4a8f-8f3e-48eceea0565d\",\"5595d1b3-c996-4098-9328-2fc8ddcfefdc\",\"a5f12354-2b66-43c7-86a8-352b2aa77c4f\",\"ad98d568-e13b-4afe-80f7-888182f28973\",\"2ac6d6ea-6e23-4c1a-b7b9-d2219d8304c5\",\"183a81ca-68c1-430e-959f-feb766076b69\",\"d0370f19-29e8-47c1-b7c8-15a1ab7a3fde\",\"0fab6fe1-af9a-479a-a697-db924bba9724\",\"813bfea7-06ec-48be-a8a9-bc4e059b1d09\",\"0b1643be-d358-43ce-84c0-64bf3ff0d326\",\"c750ec60-6786-4f92-a992-902d940d689a\",\"5b0bc2d7-52ef-475b-bee2-d98494e2e25e\",\"02c4eae0-0403-459d-ba02-8eeeccea0959\",\"c6042189-5274-49a4-a9c7-4b019b638b6e\",\"e7cf1ab0-2490-4791-86cf-618d6b44db4e\",\"8493e732-19de-42ed-a021-c7d1e78818a8\",\"3d8a2b69-ba45-4b53-96b9-15711656acf4\",\"81897ce0-499f-4429-ab2e-003e01c7e328\",\"15322509-e7a9-48e2-8d2a-31d6e807498a\",\"b0461e46-e642-4a6d-baaa-b5e12bca94eb\",\"3c70a142-e559-49ec-ae3a-3fbe765d2a7d\",\"583652d0-8c16-4ec6-bf98-474574b8147e\",\"228becdc-0423-4dab-9c28-c0c724df47f6\",\"e91c1cc9-4164-4076-ac11-d0899b28217a\",\"5eedabbf-517b-482d-85d7-186a2e6eb514\",\"0faab6f2-7e91-4d5d-b508-c6c03a77fe72\",\"209665c1-fdaa-435d-bee7-c3f646b32745\",\"9e3820e5-3cf3-4a48-8e5d-c6e78cbef3b2\",\"3e047440-704a-4f9b-8056-103e7052bdfa\",\"2d0d58e0-a9a9-46f9-b5ef-e0e5012bfcd6\",\"c6df1f41-2d2b-423d-ad95-7c0e83a40d27\",\"df5f6b88-233a-4443-b9d4-c3b1e0c0a024\",\"5fc43142-b6e8-4a3f-b1a9-8e770bf1ca93\",\"83eea200-28a3-44c7-871f-3de92fe94a68\",\"359caafa-eb5a-47c1-91b1-523f279a8550\",\"a84839f7-ce76-4fab-918f-e34ad71a1986\",\"d013bf56-1d90-4b5f-8841-cbebabf83d5f\",\"f3d70a99-3a4d-4e80-b927-097564a350f4\",\"bc85e396-1752-4bef-8b4d-8e56785c1908\",\"6e42f50b-f670-4e97-824e-0c45194b2227\",\"81736374-6f99-4045-bc9b-aedbaefae999\",\"92aec629-065a-4636-93b6-95ddadd7efaa\",\"449661ab-915f-4ebf-8a1f-380b4b46efa5\",\"e43ac88f-4ac4-4eef-8810-c83e9fc07f40\",\"d83bc056-fd86-4585-92ae-ddc599c2e682\",\"ffdc15b1-3283-43e6-8c23-809a2b8bd549\",\"acfcf0e1-63c7-483d-a203-49e33515a94f\",\"d4cb48ba-fa07-43db-b971-ca995cd19054\",\"77f5f22f-46a4-4635-b3ae-765e2456ebd8\",\"16b03267-72db-48cf-b912-e9b230d1c140\",\"1d726abe-2176-4fbb-b208-f1797c2ee21e\",\"78a78c7a-b4f0-42ac-bfa6-e41d898a9327\",\"f552d2b8-3cf2-4b9c-bd8a-9c93268f646b\",\"a7830a87-3295-4ab0-a428-824c25c33963\",\"222128e2-2178-4e88-a942-0f9fd57040f8\",\"4b61443a-d1a3-4f7d-bda6-cbc771bab3f5\",\"72106746-bfd6-476a-a761-c6fa22364be6\",\"01957a82-1173-44af-95b2-df146b067774\",\"318eecdb-771d-4bd4-8ff1-d667a8674868\",\"1d13f79e-60db-4483-89dc-cc7bbed2cb7b\",\"55c475de-3bad-4239-ba49-4b1388aeab4f\",\"bf3476c3-463d-4916-9c09-e3e01fe56c5f\",\"a163eeb8-959a-44a2-a969-5d2f88d27ab0\",\"baccd4c7-1bf2-4542-b090-f6979b09b656\"],\"reminder_category_key\":\"resubmit\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, resubmitReminderEmailJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_ny_reminder_email.json", "some-bucket")
                .get("payload");
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);
        List<TaxReturnSubmission> submissionList = new ArrayList<>();
        List<UUID> idList = new ArrayList<>();
        for (int i = 0; i <= 100; i++) {
            TaxReturn tr = TaxReturn.testObjectFactory();
            TaxReturnSubmission trs = tr.addTaxReturnSubmission();
            trs.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
            trs.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
            submissionList.add(trs);
            idList.add(tr.getId());
        }
        doReturn(submissionList).when(taxReturnSubmissionRepository).findLatestTaxReturnSubmissions(any());
        doReturn(null).when(taxReturnSubmissionRepository).saveAll(any());

        // process payload and put items into cache
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("resubmit"), 100);

        // fetch items from cache and enqueue status change emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();

        verify(confirmationService, times(1)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_RESUBMIT));
        verify(taxReturnSubmissionRepository, times(1)).findLatestTaxReturnSubmissions(any());
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("resubmit"), 50);

        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();
        verify(confirmationService, times(2)).enqueueStatusChangeEmail(any(), eq(HtmlTemplate.REMINDER_RESUBMIT));
        verify(taxReturnSubmissionRepository, times(2)).findLatestTaxReturnSubmissions(any());
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
    }

    @Test
    public void givenReminderSubmitPayloadWithDateRange_itSendsReminderSubmitEmails() {
        // Arrange: Create a ReminderEmail payload with category "submit" and an open date range and corresponding
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);

        String reminderSubmitEmailPayload =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[], \"startDate\": \"2025-01-01\", \"endDate\": \"2025-05-01\", \"reminder_category_key\":\"submit\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, reminderSubmitEmailPayload.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);

        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_reminder_submit_email.json", "some-bucket")
                .get("payload");

        // Set up mock date for the repository methods to return
        List<TaxReturn> unsubmittedTaxReturns = new ArrayList<>();
        List<SimpleTaxReturnProjection> unsubmittedTaxReturnProjections = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            TaxReturn taxReturn = TaxReturn.testObjectFactory();
            SimpleTaxReturnProjection simpleTaxReturnProjection = SimpleTaxReturnImpl.fromTaxReturn(taxReturn);

            unsubmittedTaxReturns.add(taxReturn);
            unsubmittedTaxReturnProjections.add(simpleTaxReturnProjection);
        }

        Window<SimpleTaxReturnProjection> unsubmittedTaxReturnWindow =
                Window.from(unsubmittedTaxReturnProjections, (i) -> ScrollPosition.keyset(), false);
        doReturn(unsubmittedTaxReturns).when(taxReturnRepository).findAllByTaxReturnIds(any());
        doReturn(unsubmittedTaxReturnWindow)
                .when(taxReturnRepository)
                .findByTaxYearAndSubmitTimeIsNullAndCreatedAtBetweenOrderByCreatedAtAsc(
                        any(), anyInt(), any(), any(), any());

        // Act: Kick off the handler, then send the emails
        reminderEmailHandler.handleNotificationEvent(payload);
        assertFalse(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        assertEquals(reminderEmailHandler.reminderEmailCacheService.size("submit"), 10);

        // Fetch items from the cache and send the emails
        reminderEmailHandler.fetchTaxReturnIdBatchFromCacheAndSendReminderEmails();
        verify(confirmationService, times(1)).enqueueUnsubmittedReturnsMessages(unsubmittedTaxReturns);
    }

    @Test
    public void givenReminderSubmitPayloadWithDateRange_noUnsubmittedReturns_itDoesNotSEndReminderSubmitEmails() {
        // Arrange: Create a ReminderEmail payload with category "submit" and an open date range and corresponding
        ReminderEmailHandler reminderEmailHandler =
                new ReminderEmailHandler(confirmationService, taxReturnSubmissionRepository, taxReturnRepository);

        String reminderSubmitEmailPayload =
                "{\"key\":\"reminder_email\",\"payload\":{\"ids\":[], \"startDate\": \"2025-01-01\", \"endDate\": \"2025-05-01\", \"reminder_category_key\":\"submit\"}}";
        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, reminderSubmitEmailPayload.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);

        payload = s3NotificationEventService
                .loadObjectFromS3("adhoc_job_reminder_submit_email.json", "some-bucket")
                .get("payload");

        // Set up mock date for the repository methods to return
        List<TaxReturn> unsubmittedTaxReturns = new ArrayList<>();
        List<SimpleTaxReturnProjection> unsubmittedTaxReturnProjections = new ArrayList<>();

        Window<SimpleTaxReturnProjection> unsubmittedTaxReturnWindow =
                Window.from(unsubmittedTaxReturnProjections, (i) -> ScrollPosition.keyset(), false);
        doReturn(unsubmittedTaxReturnWindow)
                .when(taxReturnRepository)
                .findByTaxYearAndSubmitTimeIsNullAndCreatedAtBetweenOrderByCreatedAtAsc(
                        any(), anyInt(), any(), any(), any());

        // Act: Kick off the handler, then send the emails
        reminderEmailHandler.handleNotificationEvent(payload);
        assertTrue(reminderEmailHandler.getNextBatchToProcess().isEmpty());
        verify(confirmationService, times(0)).enqueueUnsubmittedReturnsMessages(unsubmittedTaxReturns);
    }
}
