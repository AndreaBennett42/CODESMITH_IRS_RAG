package gov.irs.directfile.api.taxreturn.submissions;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.*;
import java.util.stream.Collectors;
import javax.xml.datatype.DatatypeConfigurationException;

import com.fasterxml.jackson.core.JsonProcessingException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

import gov.irs.factgraph.Graph;

import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.taxreturn.*;
import gov.irs.directfile.api.taxreturn.models.SubmissionEvent;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.api.util.TestDataFactory;
import gov.irs.directfile.models.FactTypeWithItem;
import gov.irs.directfile.models.LepLanguage;
import gov.irs.directfile.models.TaxReturnSubmissionReceipt;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.SendEmailQueueMessageBody;
import gov.irs.directfile.models.message.SubmissionEventFailureCategoryEnum;
import gov.irs.directfile.models.message.SubmissionEventFailureDetailEnum;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT) // Suppresses UnnecessaryStubbingException
public class ConfirmationServiceTest {

    private ConfirmationService confirmationService;

    private static final TestDataFactory testDataFactory = new TestDataFactory();

    @Mock
    private TaxReturnRepository taxReturnRepo;

    @Mock
    private SubmissionEventRepository submissionEventRepository;

    @Mock
    private TaxReturnSubmissionRepository taxReturnSubmissionRepo;

    @Mock
    private TaxReturnService taxReturnService;

    @Mock
    private SendEmailQueueService sendEmailQueueService;

    @Mock
    private FactGraphService factGraphService;

    @Mock
    private StatusResponseBodyCacheService statusResponseBodyCacheService;

    @Autowired
    private TestEntityManager entityManager;

    @BeforeEach
    void setup() {
        confirmationService = new ConfirmationService(
                taxReturnRepo,
                taxReturnSubmissionRepo,
                taxReturnService,
                sendEmailQueueService,
                factGraphService,
                submissionEventRepository,
                2,
                statusResponseBodyCacheService);
    }

    @Test
    public void handleSubmissionConfirmations_UpdatesTaxReturnSubmissions() throws DatatypeConfigurationException {
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturn tr2 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());
        trs1.setId(UUID.randomUUID());

        TaxReturnSubmission trs2 = tr2.addTaxReturnSubmission();
        trs2.setSubmissionId("submissionId2");
        trs2.setTaxReturnId(tr2.getId());
        trs2.setId(UUID.randomUUID());
        doReturn(Optional.of(tr1)).when(taxReturnRepo).findById(tr1.getId());
        doReturn(Optional.of(tr2)).when(taxReturnRepo).findById(tr2.getId());
        doReturn(Optional.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId());
        doReturn(Optional.of(trs2))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr2.getId());
        doReturn(List.of(tr1, tr2)).when(taxReturnRepo).findAllByTaxReturnIds(any());
        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());
        doReturn(List.of(trs1.getId(), trs2.getId()))
                .when(taxReturnSubmissionRepo)
                .findIdBySubmissionId(any());

        TaxReturn taxReturn1 = taxReturnRepo.findById(tr1.getId()).get();
        TaxReturn taxReturn2 = taxReturnRepo.findById(tr2.getId()).get();

        assertEquals(taxReturn1.getTaxReturnSubmissions().size(), 1);
        assertEquals(taxReturn2.getTaxReturnSubmissions().size(), 1);
        assertEquals(
                taxReturn1.getTaxReturnSubmissions().stream().findFirst().get().getSubmissionEvents().stream()
                        .findFirst()
                        .get()
                        .getEventType(),
                SubmissionEventTypeEnum.PROCESSING);
        assertEquals(
                taxReturn2.getTaxReturnSubmissions().stream().findFirst().get().getSubmissionEvents().stream()
                        .findFirst()
                        .get()
                        .getEventType(),
                SubmissionEventTypeEnum.PROCESSING);

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr2.getId(), "submissionId2", "receiptId2", new Date());

        confirmationService.handleSubmissionConfirmations(
                List.of(taxReturnSubmissionReceipt1, taxReturnSubmissionReceipt2));

        trs1 = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(taxReturn1.getId())
                .get();
        trs2 = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(taxReturn2.getId())
                .get();

        assertEquals(trs1.getSubmissionId(), taxReturnSubmissionReceipt1.getSubmissionId());
        assertEquals(trs2.getSubmissionId(), taxReturnSubmissionReceipt2.getSubmissionId());

        assertEquals(trs1.getReceiptId(), taxReturnSubmissionReceipt1.getReceiptId());
        assertEquals(trs2.getReceiptId(), taxReturnSubmissionReceipt2.getReceiptId());

        assertEquals(trs1.getSubmissionReceivedAt(), taxReturnSubmissionReceipt1.getSubmissionReceivedAt());
        assertEquals(trs2.getSubmissionReceivedAt(), taxReturnSubmissionReceipt2.getSubmissionReceivedAt());

        assertEquals(trs1.getSubmissionEvents().size(), 2);
        assertEquals(trs2.getSubmissionEvents().size(), 2);

        assertTrue(trs1.getSubmissionEvents().stream()
                .anyMatch(x -> x.getEventType().equals(SubmissionEventTypeEnum.SUBMITTED)));
        assertTrue(trs2.getSubmissionEvents().stream()
                .anyMatch(x -> x.getEventType().equals(SubmissionEventTypeEnum.SUBMITTED)));
    }

    @Test
    public void addSubmissionEvents_UpdatesTaxReturnSubmissions() throws DatatypeConfigurationException {
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturn tr2 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());
        trs1.setId(UUID.randomUUID());
        TaxReturnSubmission trs2 = tr2.addTaxReturnSubmission();
        trs2.setSubmissionId("submissionId2");
        trs2.setTaxReturnId(tr2.getId());
        trs2.setId(UUID.randomUUID());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr2.getId(), "submissionId2", "receiptId2", new Date());

        ArrayList<SubmissionStatusesMessage> message = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        List<String> rejected = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        message.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        message.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));

        doReturn(List.of(tr1, tr2)).when(taxReturnRepo).findAllByTaxReturnIds(any());

        TaxReturnSubmission finalTrs1 = trs1;
        TaxReturnSubmission finalTrs2 = trs2;
        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(
                        argThat(t -> t.containsAll(List.of(finalTrs1.getId(), finalTrs2.getId()))));
        doReturn(List.of(trs1.getId(), trs2.getId()))
                .when(taxReturnSubmissionRepo)
                .findIdBySubmissionId(any());
        doReturn(List.of(trs1)).when(taxReturnSubmissionRepo).findAllBySubmissionIds(accepted);
        doReturn(List.of(trs2)).when(taxReturnSubmissionRepo).findAllBySubmissionIds(rejected);
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(List.of(finalTrs1.getId()));
        doReturn(List.of(trs2))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(List.of(finalTrs2.getId()));
        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());

        confirmationService.handleSubmissionConfirmations(
                List.of(taxReturnSubmissionReceipt1, taxReturnSubmissionReceipt2));
        confirmationService.addSubmissionEvents(message);

        accepted.forEach(
                entry -> verify(statusResponseBodyCacheService, times(1)).clearKey(entry));
        rejected.forEach(
                entry -> verify(statusResponseBodyCacheService, times(1)).clearKey(entry));

        trs1 = taxReturnSubmissionRepo.findAllBySubmissionIds(accepted).get(0);
        trs2 = taxReturnSubmissionRepo.findAllBySubmissionIds(rejected).get(0);

        assertTrue(trs1.getSubmissionEvents().stream()
                .anyMatch(x -> x.getEventType().equals(SubmissionEventTypeEnum.ACCEPTED)));
        assertTrue(trs2.getSubmissionEvents().stream()
                .anyMatch(x -> x.getEventType().equals(SubmissionEventTypeEnum.REJECTED)));
        assertEquals(trs2.getSubmissionEvents().size(), 3);
        assertEquals(trs1.getSubmissionEvents().size(), 3);
    }

    @Test
    public void createStatusChangeMessages_CreatesMessagesCorrectly_SingleReturns()
            throws DatatypeConfigurationException {
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());

        ArrayList<SubmissionStatusesMessage> message = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        message.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));

        doReturn(List.of(trs1)).when(taxReturnSubmissionRepo).findAllBySubmissionIds(accepted);
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());

        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());

        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt1));
        Map<String, List<TaxReturnSubmission>> statusTaxReturnSubmissionMap =
                confirmationService.addSubmissionEvents(message);
        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> messagesToSend =
                confirmationService.createStatusChangeMessages(statusTaxReturnSubmissionMap);

        List<SendEmailQueueMessageBody> rejectedMessage = messagesToSend.get(HtmlTemplate.REJECTED);
        List<SendEmailQueueMessageBody> acceptedMessage = messagesToSend.get(HtmlTemplate.ACCEPTED);

        assertEquals(messagesToSend.size(), 1);

        assertNull(rejectedMessage);
        assertEquals(acceptedMessage.size(), 1);
        assertEquals(acceptedMessage.stream().findFirst().get().getTo(), "some_email@email.com");
        assertEquals(acceptedMessage.stream().findFirst().get().getLanguageCode(), LepLanguage.ENGLISH.toCode());
    }

    @Test
    public void createStatusChangeMessages_CreatesMessagesCorrectly_MultipleReturns()
            throws DatatypeConfigurationException {
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturn tr2 = TaxReturn.testObjectFactory();

        User mockUser1 = mock(User.class);
        User mockUser2 = mock(User.class);
        UUID userId1 = UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba111");
        UUID userId2 = UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba222");
        when(mockUser1.getId()).thenReturn(userId1);
        when(mockUser2.getId()).thenReturn(userId2);

        tr1.addOwner(mockUser1);
        tr2.addOwner(mockUser2);

        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());

        TaxReturnSubmission trs2 = tr2.addTaxReturnSubmission();
        trs2.setSubmissionId("submissionId2");
        trs2.setTaxReturnId(tr2.getId());

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr2.getId(), "submissionId2", "receiptId2", new Date());
        ArrayList<SubmissionStatusesMessage> message = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        List<String> rejected = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        message.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        message.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));

        doReturn(List.of(trs1, trs2)).when(taxReturnSubmissionRepo).findAllBySubmissionIds(any());
        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());
        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());

        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        confirmationService.handleSubmissionConfirmations(
                List.of(taxReturnSubmissionReceipt1, taxReturnSubmissionReceipt2));
        Map<String, List<TaxReturnSubmission>> statusTaxReturnSubmissionMap =
                confirmationService.addSubmissionEvents(message);

        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> messagesToSend =
                confirmationService.createStatusChangeMessages(statusTaxReturnSubmissionMap);

        List<SendEmailQueueMessageBody> rejectedMessage = messagesToSend.get(HtmlTemplate.REJECTED);
        List<SendEmailQueueMessageBody> acceptedMessage = messagesToSend.get(HtmlTemplate.ACCEPTED);

        assertEquals(messagesToSend.size(), 2);

        assertEquals(rejectedMessage.size(), 1);
        assertEquals(rejectedMessage.stream().findFirst().get().getTo(), "some_email@email.com");
        assertEquals(rejectedMessage.stream().findFirst().get().getLanguageCode(), LepLanguage.ENGLISH.toCode());
        assertEquals(userId2, rejectedMessage.stream().findFirst().get().getUserId());

        assertEquals(acceptedMessage.size(), 1);
        assertEquals(acceptedMessage.stream().findFirst().get().getTo(), "some_email@email.com");
        assertEquals(acceptedMessage.stream().findFirst().get().getLanguageCode(), LepLanguage.ENGLISH.toCode());
        assertEquals(userId1, acceptedMessage.stream().findFirst().get().getUserId());
    }

    @Test
    public void
            handleStatusChangeEvents_ForMultipleSubmissionStatusMessages_EnqueuesCorrectNumberOfMessagesWithCorrectFormat() {
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturn tr2 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());
        TaxReturnSubmission trs2 = tr2.addTaxReturnSubmission();
        trs2.setSubmissionId("submissionId2");
        trs2.setTaxReturnId(tr2.getId());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr2.getId(), "submissionId2", "receiptId2", new Date());
        ArrayList<SubmissionStatusesMessage> message = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        List<String> rejected = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        message.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        message.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));

        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());
        doReturn(List.of(trs1, trs2)).when(taxReturnSubmissionRepo).findAllBySubmissionIds(any());
        doReturn(List.of(trs1, trs2))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());
        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        confirmationService.handleSubmissionConfirmations(
                List.of(taxReturnSubmissionReceipt1, taxReturnSubmissionReceipt2));
        confirmationService.handleStatusChangeEvents(message);

        // Verify the format and values of the messages
        ArgumentCaptor<Map<HtmlTemplate, List<SendEmailQueueMessageBody>>> argumentCaptor =
                ArgumentCaptor.forClass(Map.class);
        verify(sendEmailQueueService, times(1)).enqueue(argumentCaptor.capture());

        Map<HtmlTemplate, List<SendEmailQueueMessageBody>> argument = argumentCaptor.getValue();

        SendEmailQueueMessageBody sendEmailQueueMessageBody1 =
                argument.get(HtmlTemplate.ACCEPTED).get(0);
        assertEquals("submissionId1", sendEmailQueueMessageBody1.getSubmissionId());
        assertEquals("some_email@email.com", sendEmailQueueMessageBody1.getTo());
        assertEquals(tr1.getId(), sendEmailQueueMessageBody1.getTaxReturnId());
        assertEquals(LepLanguage.ENGLISH.toCode(), sendEmailQueueMessageBody1.getLanguageCode());
        assertNull(sendEmailQueueMessageBody1.getEmailId());

        SendEmailQueueMessageBody sendEmailQueueMessageBody2 =
                argument.get(HtmlTemplate.REJECTED).get(0);
        assertEquals("submissionId2", sendEmailQueueMessageBody2.getSubmissionId());
        assertEquals("some_email@email.com", sendEmailQueueMessageBody2.getTo());
        assertEquals(tr2.getId(), sendEmailQueueMessageBody2.getTaxReturnId());
        assertEquals(LepLanguage.ENGLISH.toCode(), sendEmailQueueMessageBody2.getLanguageCode());
        assertNull(sendEmailQueueMessageBody1.getEmailId());
    }

    @Test
    void givenFailedSubmissions_whenHandleSubmissionsFailure_thenTaxReturnSubmissionsUpdated() {
        List<TaxReturnSubmission> taxReturnSubmissions = new ArrayList<>();
        List<TaxReturnSubmissionReceipt> taxReturnSubmissionReceipts = new ArrayList<>();
        List<SubmissionConfirmationPayloadV2Entry> entries = new ArrayList<>();
        List<TaxReturn> taxReturns = List.of(
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory());

        taxReturns.forEach(taxReturn -> {
            TaxReturnSubmission taxReturnSubmission = TaxReturnSubmission.testObjectFactory(taxReturn);
            taxReturnSubmission.setSubmissionId(UUID.randomUUID().toString());
            TaxReturnSubmissionReceipt taxReturnSubmissionReceipt = new TaxReturnSubmissionReceipt(
                    taxReturn.getId(), taxReturnSubmission.getSubmissionId(), null, null);

            taxReturnSubmissions.add(taxReturnSubmission);
            taxReturnSubmissionReceipts.add(taxReturnSubmissionReceipt);
            entries.add(new SubmissionConfirmationPayloadV2Entry(
                    taxReturnSubmissionReceipt,
                    SubmissionEventTypeEnum.FAILED,
                    Map.of(
                            "failureCategory", SubmissionEventFailureCategoryEnum.PROCESSING.getFailureCategory(),
                            "failureDetail",
                                    SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING.getFailureDetail())));
        });

        when(taxReturnSubmissionRepo.findAllBySubmissionIds(anyList())).thenReturn(taxReturnSubmissions);
        confirmationService.handleSubmissionFailures(entries);
        verify(taxReturnSubmissionRepo, times(1)).saveAll(taxReturnSubmissions);
        taxReturnSubmissions.forEach(entry -> {
            List<SubmissionEvent> failedSubmissionEventsList = entry.getSubmissionEvents().stream()
                    .filter(submissionEvent -> SubmissionEventTypeEnum.FAILED.equals(submissionEvent.getEventType()))
                    .collect(Collectors.toList());
            assertFalse(failedSubmissionEventsList.isEmpty());
            assertEquals(1, failedSubmissionEventsList.size());
            verify(statusResponseBodyCacheService, times(1)).clearKey(entry.getSubmissionId());
        });
    }

    @Test
    void givenFailedSubmissions_whenHandleSubmissionsFailure_thenDispatchReenqueued() {
        List<TaxReturnSubmission> taxReturnSubmissions = new ArrayList<>();
        List<TaxReturnSubmissionReceipt> taxReturnSubmissionReceipts = new ArrayList<>();
        List<SubmissionConfirmationPayloadV2Entry> entries = new ArrayList<>();
        List<TaxReturn> taxReturns = List.of(
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory(),
                TaxReturn.testObjectFactory());

        taxReturns.forEach(taxReturn -> {
            TaxReturnSubmission taxReturnSubmission = TaxReturnSubmission.testObjectFactory(taxReturn);
            taxReturnSubmission.setSubmissionId(UUID.randomUUID().toString());
            taxReturnSubmission.setTaxReturnId(taxReturn.getId());
            taxReturnSubmission.setTaxReturn(taxReturn);
            TaxReturnSubmissionReceipt taxReturnSubmissionReceipt = new TaxReturnSubmissionReceipt(
                    taxReturn.getId(), taxReturnSubmission.getSubmissionId(), null, null);

            taxReturnSubmissions.add(taxReturnSubmission);
            taxReturnSubmissionReceipts.add(taxReturnSubmissionReceipt);
            entries.add(new SubmissionConfirmationPayloadV2Entry(
                    taxReturnSubmissionReceipt,
                    SubmissionEventTypeEnum.FAILED,
                    Map.of(
                            "failureCategory",
                            SubmissionEventFailureCategoryEnum.PROCESSING.getFailureCategory(),
                            "failureDetail",
                            SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING.getFailureDetail())));
        });

        when(taxReturnSubmissionRepo.findAllBySubmissionIds(anyList())).thenReturn(taxReturnSubmissions);
        when(taxReturnRepo.findAllByTaxReturnIds(anyList())).thenReturn(taxReturns);
        confirmationService.handleSubmissionFailures(entries);
        verify(taxReturnService, times(5)).stubEnqueueDispatch();
    }

    @Test
    void givenFailedSubmissions_whenEnqueueDispatchForFailedSubmissions_whenFirstFailureAttempt_thenDispatchEnqueued() {
        TaxReturn taxReturn = TaxReturn.testObjectFactory();
        Map<UUID, String> taxReturnIdSubmissionIdMap = new HashMap<>();
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();

        TaxReturnSubmission taxReturnSubmission = TaxReturnSubmission.testObjectFactory(taxReturn);
        taxReturnSubmission.setSubmissionId("123456789");
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
        taxReturnSubmission.addSubmissionEvent(
                SubmissionEventTypeEnum.FAILED,
                SubmissionEventFailureCategoryEnum.PROCESSING,
                SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING);
        when(taxReturnSubmissionRepo.save(taxReturnSubmission)).thenReturn(taxReturnSubmission);
        taxReturnSubmissionRepo.save(taxReturnSubmission);
        taxReturnIdSubmissionIdMap.put(taxReturn.getId(), taxReturnSubmission.getSubmissionId());
        taxReturnTaxReturnSubmissionMap.put(taxReturn.getId(), taxReturnSubmission);

        when(submissionEventRepository.countFailedEvents(any())).thenReturn(1);
        when(taxReturnRepo.findAllByTaxReturnIds(anyList())).thenReturn(List.of(taxReturn));
        confirmationService.enqueueDispatchForFailedSubmissions(
                taxReturnIdSubmissionIdMap, taxReturnTaxReturnSubmissionMap);
        verify(taxReturnService, times(1)).stubEnqueueDispatch();
    }

    @Test
    void
            givenFailedSubmissions_whenEnqueueDispatchForFailedSubmissions_whenSecondFailureAttempt_thenDispatchEnqueued() {
        TaxReturn taxReturn = TaxReturn.testObjectFactory();
        Map<UUID, String> taxReturnIdSubmissionIdMap = new HashMap<>();
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();

        TaxReturnSubmission taxReturnSubmission = TaxReturnSubmission.testObjectFactory(taxReturn);
        taxReturnSubmission.setSubmissionId("123456789");
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
        for (int i = 0; i <= 2; i++) {
            taxReturnSubmission.addSubmissionEvent(
                    SubmissionEventTypeEnum.FAILED,
                    SubmissionEventFailureCategoryEnum.PROCESSING,
                    SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING);
        }
        when(taxReturnSubmissionRepo.save(taxReturnSubmission)).thenReturn(taxReturnSubmission);
        taxReturnSubmissionRepo.save(taxReturnSubmission);
        taxReturnIdSubmissionIdMap.put(taxReturn.getId(), taxReturnSubmission.getSubmissionId());
        taxReturnTaxReturnSubmissionMap.put(taxReturn.getId(), taxReturnSubmission);

        when(submissionEventRepository.countFailedEvents(any())).thenReturn(2);
        when(taxReturnRepo.findAllByTaxReturnIds(anyList())).thenReturn(List.of(taxReturn));
        confirmationService.enqueueDispatchForFailedSubmissions(
                taxReturnIdSubmissionIdMap, taxReturnTaxReturnSubmissionMap);
        verify(taxReturnService, times(1)).stubEnqueueDispatch();
    }

    @Test
    void
            givenFailedSubmissions_whenEnqueueDispatchForFailedSubmissions_whenThirdFailureAttempt_thenDispatchNotEnqueued()
                    throws JsonProcessingException {
        TaxReturn taxReturn = TaxReturn.testObjectFactory();
        Map<String, FactTypeWithItem> facts = testDataFactory.auditedFacts();
        taxReturn.setFacts(facts);
        Map<UUID, String> taxReturnIdSubmissionIdMap = new HashMap<>();
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();

        TaxReturnSubmission taxReturnSubmission = TaxReturnSubmission.testObjectFactory(taxReturn);
        taxReturnSubmission.setFacts(facts);
        taxReturnSubmission.setSubmissionId("123456789");
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
        for (int i = 0; i <= 3; i++) {
            taxReturnSubmission.addSubmissionEvent(
                    SubmissionEventTypeEnum.FAILED,
                    SubmissionEventFailureCategoryEnum.PROCESSING,
                    SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING);
        }
        when(taxReturnSubmissionRepo.save(taxReturnSubmission)).thenReturn(taxReturnSubmission);
        taxReturnSubmissionRepo.save(taxReturnSubmission);
        taxReturnIdSubmissionIdMap.put(taxReturn.getId(), taxReturnSubmission.getSubmissionId());
        taxReturnTaxReturnSubmissionMap.put(taxReturn.getId(), taxReturnSubmission);
        when(submissionEventRepository.countFailedEvents(any())).thenReturn(3);
        when(taxReturnRepo.findAllByTaxReturnIds(anyList())).thenReturn(List.of(taxReturn));

        confirmationService.enqueueDispatchForFailedSubmissions(
                taxReturnIdSubmissionIdMap, taxReturnTaxReturnSubmissionMap);
        verify(taxReturnService, times(0)).stubEnqueueDispatch();
        verify(sendEmailQueueService, times(1)).enqueue(any());
    }

    @Test
    void
            givenFailedSubmissions_whenEnqueueDispatchForFailedSubmissions_whenSomeSubmissionsAreAbove2FailedAttemptsAndOthersAreNot_thenDispatchEnqueued_thenEmailAlsoEnqueued()
                    throws JsonProcessingException {
        TaxReturn taxReturn = TaxReturn.testObjectFactory();
        TaxReturn taxReturn2 = TaxReturn.testObjectFactory();
        Map<String, FactTypeWithItem> facts = testDataFactory.auditedFacts();
        Map<String, FactTypeWithItem> facts2 = testDataFactory.auditedFacts();
        taxReturn.setFacts(facts);
        taxReturn2.setFacts(facts2);
        Map<UUID, String> taxReturnIdSubmissionIdMap = new HashMap<>();
        Map<UUID, String> taxReturnIdSubmissionIdMap2 = new HashMap<>();
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap2 = new HashMap<>();

        TaxReturnSubmission taxReturnSubmission = TaxReturnSubmission.testObjectFactory(taxReturn);
        taxReturnSubmission.setFacts(facts);
        taxReturnSubmission.setSubmissionId("123456789");
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);

        TaxReturnSubmission taxReturnSubmission2 = TaxReturnSubmission.testObjectFactory(taxReturn2);
        taxReturnSubmission2.setFacts(facts2);
        taxReturnSubmission2.setSubmissionId("99999999");
        taxReturnSubmission2.addSubmissionEvent(SubmissionEventTypeEnum.PROCESSING);
        for (int i = 0; i <= 3; i++) {
            taxReturnSubmission2.addSubmissionEvent(
                    SubmissionEventTypeEnum.FAILED,
                    SubmissionEventFailureCategoryEnum.PROCESSING,
                    SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING);
        }
        taxReturnSubmission.addSubmissionEvent(
                SubmissionEventTypeEnum.FAILED,
                SubmissionEventFailureCategoryEnum.PROCESSING,
                SubmissionEventFailureDetailEnum.SUBMISSION_PROCESSING);
        when(taxReturnSubmissionRepo.save(taxReturnSubmission)).thenReturn(taxReturnSubmission);
        when(taxReturnSubmissionRepo.save(taxReturnSubmission2)).thenReturn(taxReturnSubmission2);
        taxReturnSubmissionRepo.save(taxReturnSubmission);
        taxReturnSubmissionRepo.save(taxReturnSubmission2);
        taxReturnIdSubmissionIdMap.put(taxReturn.getId(), taxReturnSubmission.getSubmissionId());
        taxReturnIdSubmissionIdMap2.put(taxReturn2.getId(), taxReturnSubmission2.getSubmissionId());
        taxReturnTaxReturnSubmissionMap.put(taxReturn.getId(), taxReturnSubmission);
        taxReturnTaxReturnSubmissionMap2.put(taxReturn2.getId(), taxReturnSubmission2);
        when(submissionEventRepository.countFailedEvents(taxReturnSubmission.getId()))
                .thenReturn(1);
        when(taxReturnRepo.findAllByTaxReturnIds(anyList())).thenReturn(List.of(taxReturn));

        confirmationService.enqueueDispatchForFailedSubmissions(
                taxReturnIdSubmissionIdMap, taxReturnTaxReturnSubmissionMap);
        verify(taxReturnService, times(1)).stubEnqueueDispatch();
        verify(sendEmailQueueService, times(0)).enqueue(any());

        when(taxReturnRepo.findAllByTaxReturnIds(anyList())).thenReturn(List.of(taxReturn2));
        when(submissionEventRepository.countFailedEvents(taxReturnSubmission2.getId()))
                .thenReturn(3);
        confirmationService.enqueueDispatchForFailedSubmissions(
                taxReturnIdSubmissionIdMap2, taxReturnTaxReturnSubmissionMap2);
        verify(sendEmailQueueService, times(1)).enqueue(any());
        verify(taxReturnService, times(1)).stubEnqueueDispatch();
    }

    @Test
    // Case where Submit App sends a second confirmation for an already submitted return
    public void itIgnoresNewConfirmationEventWhenItAlreadyExistsForTaxReturn() {
        /**
         * Arrange:
         *
         * Create a tax return, and tax return submission. Add processing and submitted event to the tax return Submission
         * */
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());

        TaxReturnSubmission trs2 = TaxReturnSubmission.testObjectFactory(tr1);
        trs2.setSubmissionId("submissionId1");
        trs2.setTaxReturnId(tr1.getId());

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId2", new Date());

        doReturn(List.of(tr1)).when(taxReturnRepo).findAllByTaxReturnIds(any());
        doReturn(Optional.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(
                        List.of(trs1.getTaxReturn().getId()));

        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt1));
        /**
         *
         * Act: Attempt to add another "submitted" event to the TaxReturnSubmission through the confirmationService
         * */
        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt2));

        /**
         *
         * Assert: Expect that the tax return submission only has 1 submitted event. And the submission id matches the first receipt id.
         * */
        TaxReturnSubmission persistedSubmission = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId())
                .get();
        assertEquals("receiptId1", persistedSubmission.getReceiptId());
        List<SubmissionEvent> submittedEvents = persistedSubmission.getSubmissionEvents().stream()
                .filter(submissionEvent -> SubmissionEventTypeEnum.SUBMITTED.equals(submissionEvent.getEventType()))
                .toList();

        assertEquals(1, submittedEvents.size());
    }

    @Test
    public void itWritesAcceptedSubmissionEventIfItExistsWhenDuplicateSubmissionIdsForSameTaxReturn() {
        /**
         *
         * Arrange: Create a tax return + tax return submission. Add processing + submitted events
         * */
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId2", new Date());

        doReturn(List.of(tr1)).when(taxReturnRepo).findAllByTaxReturnIds(any());
        doReturn(Optional.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());

        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt1));

        /**
         *
         * Act: Call handleStatusChanges, including two status change events for the return
         * One accepted, one rejected.
         *
         * This simulates case where we may have submitted the same return + submission id twice
         * */
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());
        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        ArrayList<SubmissionStatusesMessage> message = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        List<String> rejected = new ArrayList<>();
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        message.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        message.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));
        confirmationService.handleStatusChangeEvents(message);

        /**
         *
         * Assert:
         * Expect that an 'ACCEPTED' submission event was written to the database because
         * there was an 'ACCEPTED' status among the multiple submission statuses for the return.
         *
         * Expect that there are no rejected events for this tax return submission.
         * */
        TaxReturnSubmission persistedSubmission = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId())
                .get();

        boolean submissionHasRejectedEvent = persistedSubmission.getSubmissionEvents().stream()
                .anyMatch(x -> SubmissionEventTypeEnum.REJECTED.equals(x.getEventType()));

        assertFalse(submissionHasRejectedEvent);
        boolean submissionHasAcceptedEvent = persistedSubmission.getSubmissionEvents().stream()
                .anyMatch(x -> SubmissionEventTypeEnum.ACCEPTED.equals(x.getEventType()));
        assertTrue(submissionHasAcceptedEvent);
    }

    @Test
    public void itWritesRejectedSubmissionEventIfItExistsWhenDuplicateRejectedStatusForSameTaxReturn() {
        /**
         *
         * Arrange: Create a tax return + tax return submission. Add processing + submitted events
         * */
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId2", new Date());

        doReturn(List.of(tr1)).when(taxReturnRepo).findAllByTaxReturnIds(any());
        doReturn(Optional.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());

        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt1));

        /**
         *
         * Act: Call handleStatusChanges, including two status change events for the return
         * 2 rejections for the same return.
         *
         * This simulates case where we may have submitted the sa.me return + submission id twice
         * and the initial submission to MeF was rejected
         * */
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());
        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        ArrayList<SubmissionStatusesMessage> message = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        List<String> rejected = new ArrayList<>();
        // Add one rejection to list to simulate initial rejection due to some issue with the tax return
        rejected.add(taxReturnSubmissionReceipt1.getSubmissionId());
        // Add another rejection to list to simulate a second rejection due to the return being submitted a second time
        // with the same id
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        message.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        message.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));
        confirmationService.handleStatusChangeEvents(message);

        /**
         *
         * Assert:
         * Expect that an 'REJECTED' submission event was written to the database because
         * there were two 'REJECTED' statuses among the submission statuses.
         *
         * Expect that there is 1 rejected events for this tax return submission. We only record
         * the initial rejection in the backend application itself because we're using
         * Submission event to understand the current status of a return.
         * */
        TaxReturnSubmission persistedSubmission = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId())
                .get();

        long submissionRejectedEventCount = persistedSubmission.getSubmissionEvents().stream()
                .filter(x -> SubmissionEventTypeEnum.REJECTED.equals(x.getEventType()))
                .count();

        assertEquals(1, submissionRejectedEventCount);
    }

    @Test
    public void itRespectsExistingAcceptedEventForTaxReturnSubmissionWhenAnotherRejectedEventIsReceived() {
        // Arrange: Create a Tax Return with a 'submitted' event + an 'accepted' event
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId2", new Date());

        doReturn(List.of(tr1)).when(taxReturnRepo).findAllByTaxReturnIds(any());
        doReturn(Optional.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithTerminalEventsByTaxReturnSubmissionIdByEventType(any(), any());

        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt1));

        // Act: call handleStatusEvents with a new 'rejected' event for the tax return, after we already have an
        // 'accepted' status event recorded
        ArrayList<SubmissionStatusesMessage> acceptedStatusChangeMessage = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        acceptedStatusChangeMessage.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        confirmationService.handleStatusChangeEvents(acceptedStatusChangeMessage);

        ArrayList<SubmissionStatusesMessage> rejectedStatusChangeMessage = new ArrayList<>();
        List<String> rejected = new ArrayList<>();
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        rejectedStatusChangeMessage.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));
        confirmationService.handleStatusChangeEvents(rejectedStatusChangeMessage);

        // Assert: expect that we did not record a 'rejected' event for the tax return because we already accepted this
        // return. The tax return has an accepted
        // event and no rejected events
        TaxReturnSubmission persistedSubmission = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId())
                .get();

        long submissionRejectedEventCount = persistedSubmission.getSubmissionEvents().stream()
                .filter(x -> SubmissionEventTypeEnum.REJECTED.equals(x.getEventType()))
                .count();
        long submissionAcceptedEventCount = persistedSubmission.getSubmissionEvents().stream()
                .filter(x -> SubmissionEventTypeEnum.ACCEPTED.equals(x.getEventType()))
                .count();

        assertEquals(0, submissionRejectedEventCount);
        assertEquals(1, submissionAcceptedEventCount);
    }

    @Test
    // Test case assuming we receive out of order events for a return. Meaning we get a rejected status change event
    // before receiving an
    // accepted status change event for the same return
    public void itWritesAcceptedEventIfExistingTaxReturnSubmissionEventIsARejectedEvent() {
        // Arrange: Create a TaxReturn + 'submitted' and 'rejected' event
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        TaxReturnSubmission trs1 = tr1.addTaxReturnSubmission();
        trs1.setSubmissionId("submissionId1");
        trs1.setTaxReturnId(tr1.getId());

        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());
        TaxReturnSubmissionReceipt taxReturnSubmissionReceipt2 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId2", new Date());

        doReturn(List.of(tr1)).when(taxReturnRepo).findAllByTaxReturnIds(any());
        doReturn(Optional.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId());
        doReturn(List.of(trs1)).when(taxReturnSubmissionRepo).findAllBySubmissionIds(any());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(any());
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());

        // Call handleSubmissionConfirmation for taxReturnSubmissionReceipt2, simulating that our system only received
        // the second receipt
        confirmationService.handleSubmissionConfirmations(List.of(taxReturnSubmissionReceipt2));
        // Act: Send a new 'rejected' status change event, then send an 'accepted' status change event to simulate out
        // of order events

        // Act I: Send a 'rejected' status change event
        doReturn("some_email@email.com").when(taxReturnService).getFactGraphEmail(any());

        Graph graph = mock(Graph.class);
        doReturn(graph).when(factGraphService).getGraph(anyMap());

        ArrayList<SubmissionStatusesMessage> rejectedStatusChangeMessage = new ArrayList<>();
        List<String> rejected = new ArrayList<>();
        rejected.add(taxReturnSubmissionReceipt2.getSubmissionId());
        rejectedStatusChangeMessage.add(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, rejected));
        confirmationService.handleStatusChangeEvents(rejectedStatusChangeMessage);

        // Act II: Send an 'accepted' status change event
        doReturn(List.of(trs1))
                .when(taxReturnSubmissionRepo)
                .findAllWithTerminalEventsByTaxReturnSubmissionIdByEventType(any(), any());
        doReturn(List.of()).when(taxReturnSubmissionRepo).findAllWithoutTerminalEventsByTaxReturnSubmissionId(any());
        ArrayList<SubmissionStatusesMessage> acceptedStatusChangeMessage = new ArrayList<>();
        List<String> accepted = new ArrayList<>();
        accepted.add(taxReturnSubmissionReceipt1.getSubmissionId());
        acceptedStatusChangeMessage.add(new SubmissionStatusesMessage(HtmlTemplate.ACCEPTED, accepted));
        confirmationService.handleStatusChangeEvents(acceptedStatusChangeMessage);

        // Assert: Expect that the 'accepted' event is written to the database. The tax return submission should have an
        // accepted event even though we ack-ed a rejection status change already

        // saveAll should be called 3 times in this scenario: 1 for the submission confirmation, 1 for the rejected
        // status change and 1 for the accepted status change
        verify(taxReturnSubmissionRepo, times(3)).saveAll(anyList());
        verify(taxReturnSubmissionRepo, times(3)).saveAll(List.of(trs1));

        TaxReturnSubmission persistedSubmission = taxReturnSubmissionRepo
                .findLatestTaxReturnSubmissionByTaxReturnId(tr1.getId())
                .get();

        long submissionRejectedEventCount = persistedSubmission.getSubmissionEvents().stream()
                .filter(x -> SubmissionEventTypeEnum.REJECTED.equals(x.getEventType()))
                .count();
        long submissionAcceptedEventCount = persistedSubmission.getSubmissionEvents().stream()
                .filter(x -> SubmissionEventTypeEnum.ACCEPTED.equals(x.getEventType()))
                .count();

        assertEquals(1, submissionAcceptedEventCount);
        assertEquals(1, submissionRejectedEventCount);
    }

    @Test
    public void itPersistsTaxReturnSubmissionReceiptForResubmittedReturn() {
        // Arrange:
        TaxReturn tr1 = TaxReturn.testObjectFactory();
        when(taxReturnRepo.findAllByTaxReturnIds(List.of(tr1.getId()))).thenReturn(List.of(tr1));
        // a. "Submit" return that's rejected
        TaxReturnSubmission rejectedSubmission = tr1.addTaxReturnSubmission();
        rejectedSubmission.setSubmissionId("submissionId1");
        rejectedSubmission.setTaxReturnId(tr1.getId());
        rejectedSubmission.setId(UUID.randomUUID());

        TaxReturnSubmissionReceipt rejectedTaxReturnSubmissionReceipt1 =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId1", "receiptId1", new Date());

        when(taxReturnSubmissionRepo.findAllBySubmissionIds(List.of("submissionId1")))
                .thenReturn(List.of(rejectedSubmission));
        doReturn(List.of(rejectedSubmission))
                .when(taxReturnSubmissionRepo)
                .findAllWithoutTerminalEventsByTaxReturnSubmissionId(List.of(rejectedSubmission.getId()));
        doReturn(List.of(rejectedSubmission.getId()))
                .when(taxReturnSubmissionRepo)
                .findIdBySubmissionId(any());
        doReturn(List.of(rejectedSubmission))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(
                        List.of(rejectedSubmission.getTaxReturn().getId()));
        confirmationService.handleSubmissionConfirmations(List.of(rejectedTaxReturnSubmissionReceipt1));
        confirmationService.handleStatusChangeEvents(
                List.of(new SubmissionStatusesMessage(HtmlTemplate.REJECTED, List.of("submissionId1"))));

        // 2. Submit return that's accepted (and set createdAt to some time in the future to simulate editing +
        // resubmitting a return)
        TaxReturnSubmission acceptedResubmission = tr1.addTaxReturnSubmission();
        Date nextDay = Date.from(
                LocalDate.now().plusDays(1).atStartOfDay(ZoneId.systemDefault()).toInstant());
        acceptedResubmission.setCreatedAt(nextDay);
        acceptedResubmission.setTaxReturnId(tr1.getId());
        TaxReturnSubmissionReceipt acceptedTaxReturnSubmissionReceipt =
                new TaxReturnSubmissionReceipt(tr1.getId(), "submissionId2", "receiptId2", new Date());
        acceptedTaxReturnSubmissionReceipt.setSubmissionId("submissionId2");
        doReturn(List.of(acceptedResubmission))
                .when(taxReturnSubmissionRepo)
                .findLatestTaxReturnSubmissionsWithoutSubmittedSubmissionEvents(
                        List.of(acceptedResubmission.getTaxReturn().getId()));

        // Validate the receipt id is null before we call submission confirmation
        assertNull(acceptedResubmission.getReceiptId());
        confirmationService.handleSubmissionConfirmations(List.of(acceptedTaxReturnSubmissionReceipt));

        // 3. Assert that the resubmission has the receipt id populated because handleSubmissionConfirmation was able to
        // populate
        // the receiptId for the resubmission
        assertEquals("receiptId2", acceptedResubmission.getReceiptId());
    }
}
