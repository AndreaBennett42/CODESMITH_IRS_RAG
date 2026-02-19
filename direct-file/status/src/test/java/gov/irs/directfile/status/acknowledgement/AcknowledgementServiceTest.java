package gov.irs.directfile.status.acknowledgement;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

import jakarta.persistence.EntityNotFoundException;
import org.apache.commons.lang3.RandomStringUtils;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.mock.mockito.MockBean;

import gov.irs.a2a.mef.mefheader.TestCdType;
import gov.irs.mef.exception.ServiceException;
import gov.irs.mef.exception.ToolkitException;
import gov.irs.mef.services.ServiceContext;
import gov.irs.mef.services.transmitter.mtom.GetAcksResult;

import gov.irs.directfile.models.RejectedStatus;
import gov.irs.directfile.status.acknowledgement.domain.Status;
import gov.irs.directfile.status.config.StatusProperties;
import gov.irs.directfile.status.domain.*;
import gov.irs.directfile.status.domain.Error;
import gov.irs.directfile.status.error.ErrorRepository;
import gov.irs.directfile.status.error.ToolkitErrorRepository;
import gov.irs.directfile.status.mef.client.MeFAcksMTOMClientService;
import gov.irs.directfile.status.mef.client.MeFLoginClientService;
import gov.irs.directfile.status.mef.client.MeFLogoutClientService;
import gov.irs.directfile.status.mef.client.ServiceContextWrapper;
import gov.irs.directfile.status.repository.PodIdentifierRepository;
import gov.irs.directfile.status.services.StatusChangeMessageService;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anySet;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.*;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@EnableConfigurationProperties(StatusProperties.class)
@ImportAutoConfiguration(classes = SecurityAutoConfiguration.class)
@DataJpaTest
@ExtendWith(MockitoExtension.class)
class AcknowledgementServiceTest {
    @BeforeEach
    void setUp() throws ServiceException, ToolkitException {
        when(mockLoginClientService.loginWithDefaultAsid()).thenReturn(new ServiceContextWrapper(null));

        acknowledgementService = new AcknowledgementService(
                completedRepo,
                pendingRepo,
                taxReturnSubmissionRepository,
                errorRepository,
                toolkitErrorRepo,
                statusProperties,
                statusChangeMessageService,
                mockGetAcksClientService,
                mockLoginClientService,
                mockLogoutClientService,
                podIdentifierRepository);
    }

    @BeforeAll
    public static void setupSystemProperties() {
        String userDirectory = System.getProperty("user.dir");
        System.setProperty("A2A_TOOLKIT_HOME", userDirectory + "/src/test/resources/");
    }

    @AfterAll
    public static void cleanupSystemProperties() {
        System.clearProperty("A2A_TOOLKIT_HOME");
    }

    String ASID = "asid1235";

    @Autowired
    CompletedAcknowledgementRepository completedRepo;

    @Autowired
    ErrorRepository errorRepository;

    @Autowired
    PendingAcknowledgementRepository pendingRepo;

    @Autowired
    TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Autowired
    ToolkitErrorRepository toolkitErrorRepo;

    @Autowired
    PodIdentifierRepository podIdentifierRepository;

    @Autowired
    StatusProperties statusProperties;

    @MockBean
    private StatusChangeMessageService statusChangeMessageService;

    @MockBean
    MeFAcksMTOMClientService mockGetAcksClientService;

    @MockBean
    MeFLoginClientService mockLoginClientService;

    @MockBean
    MeFLogoutClientService mockLogoutClientService;

    AcknowledgementService acknowledgementService;

    public PodIdentifier createPodIdentifer(String region, String asid, int index) {
        PodIdentifier p = new PodIdentifier();
        p.setAsid(asid);
        p.setRegion(region);
        p.setPodId("dfsys-mef-status-deployment-" + index + "-" + region);
        return p;
    }

    @Test
    void GetAcknowledgementWithACompleted() {
        String submissionId = "1234562023021500001";
        UUID taxReturnId = UUID.randomUUID();
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, submissionId));

        Completed c = new Completed();
        c.setSubmissionId(submissionId);
        c.setStatus("Accepted");
        completedRepo.save(c);

        var accepted = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Accepted, accepted.getStatus());
    }

    @Test
    void GetAcknowledgementWithARejection() {
        Completed c = new Completed();
        c.setSubmissionId("12345620230215000001");
        c.setStatus("Rejected");
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation.reject.XML-123-4567-006");
        e.setMefErrorCategory("Reject and Stop");
        c.setErrors(List.of(e));
        errorRepository.save(e);
        completedRepo.save(c);

        String submissionId = "12345620230215000001";
        UUID taxReturnId = UUID.randomUUID();
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, submissionId));

        var rejected = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Rejected, rejected.getStatus());
        assertEquals(1, rejected.getRejectionCodes().size());
        assertEquals("XML-123-4567-006", rejected.getRejectionCodes().getFirst().MeFErrorCode);
        assertEquals("You messed up!", rejected.getRejectionCodes().getFirst().MeFDescription);
    }

    @Test
    void GetAcknowledgementWithMultipleRejections() {
        Completed c = new Completed();
        c.setSubmissionId("12345620230215000001");
        c.setStatus("Rejected");
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e.setMefErrorCategory("Reject");
        Error e2 = new Error();
        e2.setMefErrorCode("REJC-00001");
        e2.setErrorMessage("This was a huge problem");
        e2.setErrorCodeTranslationKey("translation/reject/REJC-00001");
        e2.setMefErrorCategory("Reject");
        c.setErrors(List.of(e, e2));
        errorRepository.saveAll(List.of(e, e2));
        completedRepo.save(c);

        String submissionId = "12345620230215000001";
        UUID taxReturnId = UUID.randomUUID();
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, submissionId));

        var rejected = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Rejected, rejected.getStatus());
        assertEquals(2, rejected.getRejectionCodes().size());
        assertEquals("XML-123-4567-006", rejected.getRejectionCodes().get(0).MeFErrorCode);
        assertEquals("You messed up!", rejected.getRejectionCodes().get(0).MeFDescription);
        assertEquals("REJC-00001", rejected.getRejectionCodes().get(1).MeFErrorCode);
        assertEquals("This was a huge problem", rejected.getRejectionCodes().get(1).MeFDescription);
    }

    @Test
    @Disabled("This test has been flaking")
    void GetAcknowledgementWithRejectionAndAcceptedReturnsLatestCompleted() {
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "1234562023021500001";

        Completed c = new Completed();
        c.setSubmissionId(submissionId);
        c.setStatus("Rejected");
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation.reject.XML-123-4567-006");
        e.setMefErrorCategory("Reject and Stop");
        c.setErrors(List.of(e));
        errorRepository.save(e);
        completedRepo.save(c);

        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, submissionId));

        submissionId = "1234562023021500002";
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, submissionId));

        Completed c1 = new Completed();
        c1.setSubmissionId(submissionId);
        c1.setStatus("Accepted");
        completedRepo.save(c1);

        var accepted = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Accepted, accepted.getStatus());
    }

    @Test
    void
            GetAcknowledgementWithNoTaxReturnSubmissionReturnsPendingIfStatusEndpointReturnsPendingByDefaultEnabledIsTrue() {
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "1234562023021500001";

        Completed c = new Completed();
        c.setSubmissionId(submissionId);
        c.setStatus("Rejected");
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation.reject.XML-123-4567-006");
        e.setMefErrorCategory("Reject and Stop");
        c.setErrors(List.of(e));
        errorRepository.save(e);
        completedRepo.save(c);

        var result = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Pending, result.getStatus());
    }

    @Test
    void GetAcknowledgementWithPending() {
        UUID taxReturnId = UUID.randomUUID();

        Pending p = new Pending();
        p.setSubmissionId("12345620230215000001");
        pendingRepo.save(p);

        var pending = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Pending, pending.getStatus());
    }

    @Test
    void
            GetAcknowledgementWithNoPreviousLookup_ifCreatePendingUponLookupIsTrue_ReturnsPendingStatusButDoesNotSaveObjectInDB() {
        UUID taxReturnId = UUID.fromString("e111c2c4-5de7-465d-8a00-b0e80caeedbc");
        var nothing = acknowledgementService.GetAcknowledgement(taxReturnId);
        assertEquals(Status.Pending, nothing.getStatus());
    }

    List<Set<String>> createBatchSubmissionIds(List<Pending> argument)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Method method = AcknowledgementService.class.getDeclaredMethod("batchPendings", Iterable.class);
        method.setAccessible(true);
        return (List<Set<String>>) method.invoke(acknowledgementService, argument);
    }

    @Test
    void batchSubmissionIdsReturnsASingleBatchForOneSubmissionId() {
        try {
            Pending p = new Pending();
            p.setSubmissionId("12345620230215000001");
            var test = createBatchSubmissionIds(List.of(p));
            assertEquals(1, test.size());
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void batchSubmissionIdsReturns10BatchesFor1000Ids() {
        try {
            List<Pending> pendings = new ArrayList<>();
            for (int i = 0; i < 1000; i++) {
                Pending p = new Pending();
                p.setSubmissionId("12345620230215000" + i);
                pendings.add(p);
            }
            var test = createBatchSubmissionIds(pendings);
            assertEquals(10, test.size());
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void batchSubmissionIdsReturns2BatchesFor101Ids() {
        try {
            List<Pending> pendings = new ArrayList<>();
            for (int i = 0; i < 101; i++) {
                Pending p = new Pending();
                p.setSubmissionId("12345620230215000" + i);
                pendings.add(p);
            }
            var test = createBatchSubmissionIds(pendings);
            assertEquals(2, test.size());
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void batchSubmissionIdsReturns1BatchFor0() {
        try {
            List<Pending> pendings = new ArrayList<>();
            var test = createBatchSubmissionIds(pendings);
            assertEquals(1, test.size());
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void CreateToolkitErrorDeletesPendingRecordForSubmissionId() {
        String submissionId = "12345678900987654321";
        Pending p = new Pending();
        p.setSubmissionId(submissionId);
        pendingRepo.save(p);
        ToolkitException tke = new ToolkitException("invalid request parameters");
        acknowledgementService.createToolkitError(p, tke);
        Optional<Pending> deletedPendingRecord = pendingRepo.findById(submissionId);
        assertEquals(Optional.empty(), deletedPendingRecord);
    }

    @Test
    void CreateToolkitErrorCreatesTheRecordWithTheSubmissionIdAndError() {
        String submissionId = "21423423423424423234";
        Pending p = new Pending();
        p.setSubmissionId(submissionId);
        p.setPodId(getDefaultPodIdentifierFromProperties(0, "us-gov-east-1"));
        pendingRepo.save(p);

        ToolkitException tke = new ToolkitException("invalid request parameters");
        acknowledgementService.createToolkitError(p, tke);

        Optional<ToolkitError> toolkitErrorRecord = toolkitErrorRepo.findById(submissionId);

        assertEquals(submissionId, toolkitErrorRecord.get().getSubmissionId());
        assertEquals(
                "gov.irs.mef.exception.ToolkitException",
                toolkitErrorRecord.get().getErrorName());
        assertEquals(
                "gov.irs.mef.exception.ToolkitException: invalid request parameters",
                toolkitErrorRecord.get().getErrorMessage());
    }

    @Test
    @Disabled
    void lookupSubmissionsAddsToolkitErrorRecordsOnToolkitExceptions() throws ServiceException, ToolkitException {
        MockitoAnnotations.openMocks(this);
        String meFClientErrorString = "MeFClientSDK000039: Invalid request parameters";
        pendingRepo.save(new Pending("12345"));
        ToolkitException tke = new ToolkitException(meFClientErrorString);
        when(mockGetAcksClientService.getAcks(isNull(), anySet())).thenThrow(tke);
        acknowledgementService.LookupSubmissions();

        Optional<Pending> pendingRecord = pendingRepo.findById("12345");
        Optional<ToolkitError> toolkitError = toolkitErrorRepo.findById("12345");
        assertEquals(Optional.empty(), pendingRecord);
        assertEquals(
                "gov.irs.mef.exception.ToolkitException", toolkitError.get().getErrorName());
        assertEquals(
                "gov.irs.mef.exception.ToolkitException: " + meFClientErrorString,
                toolkitError.get().getErrorMessage());
    }

    @Test
    @Disabled
    void lookupSubmissionsHandlesToolkitExceptions() throws ServiceException, ToolkitException {
        MockitoAnnotations.openMocks(this);
        for (int i = 0; i < 100; i++) {
            String genString = RandomStringUtils.random(20, true, true);
            Pending p = new Pending(genString);
            pendingRepo.save(p);
        }
        assertEquals(100, pendingRepo.count());

        ToolkitException tke = new ToolkitException("Invalid request parameters");

        GetAcksResult acksResult = mock(GetAcksResult.class);

        // We do this to mimic the expected behavior of the MeF client SDK on batch calls that have an issue with one
        // piece of data in the batch.
        // throw ToolkitException 7 times on a batch of size 100
        when(mockGetAcksClientService.getAcks(isNull(), anySet()))
                .thenThrow(tke)
                .thenThrow(tke)
                .thenThrow(tke)
                .thenThrow(tke)
                .thenThrow(tke)
                .thenThrow(tke)
                .thenThrow(tke)
                .thenThrow(tke)
                .thenReturn(new GetAcksResultWrapper(new AcknowledgementsListWrapper(new ArrayList<>())));

        acknowledgementService.LookupSubmissions();

        // This is to test the expected binary search pattern
        verify(mockGetAcksClientService, times(15)).getAcks(isNull(), anySet());

        // since we threw ToolkitException 7 times in a batch of size 100, we only expect one pending record to be
        // deleted and then created as a toolkit error record in the database,
        // because we use binary search pattern to divide the set until we find the single cause of the exception (100/2
        // 50/2 25/2 13/2 7/2 3/2 2/2 1/1)
        assertEquals(1, toolkitErrorRepo.count());
        assertEquals(99, pendingRepo.count());
    }

    private String getDefaultPodIdentifierFromProperties(int index, String region) {
        return "dfsys-mef-status-deployment-" + index + "-" + region;
    }

    Map<String, List<String>> getStatusSubmissionIdMap() {
        PodIdentifier pi = createPodIdentifer("us-gov-east-1", statusProperties.getAsid(), 0);
        podIdentifierRepository.save(pi);

        Map<String, List<String>> statusSubmissionIdMap = acknowledgementService.createStatusSubmissionIdMap();
        List<String> statuses = new ArrayList<>();
        statuses.add("accepted");
        statuses.add("rejected");
        statuses.add("pending");
        statuses.forEach(status -> {
            String submissionId = UUID.randomUUID().toString().substring(0, 20);
            if (Objects.equals(status, "pending")) {
                Pending p = new Pending();
                p.setSubmissionId(submissionId);
                p.setPodId(pi.getPodId());
                pendingRepo.save(p);
            }
            acknowledgementService.addToSubmissionIdMap(status, submissionId, statusSubmissionIdMap);
        });
        return statusSubmissionIdMap;
    }

    private ValidationMapCompleted getValidationErrorMap() {
        Map<String, List<String>> statusSubmissionIdMap = getStatusSubmissionIdMap();
        Map<String, List<List<String>>> validationErrorMap = new HashMap<>();
        statusSubmissionIdMap.forEach((status, submissionIdList) -> {
            if (status.equals("rejected")) {
                String submissionId = submissionIdList.get(0);
                validationErrorMap.put(submissionId, new ArrayList<>());
                List<String> errorList1 = new ArrayList<>();
                List<String> errorList2 = new ArrayList<>();
                String ruleNum = "R0000-904-03";
                String severityCd = "Reject and Stop";
                String errorMessageTxt =
                        "Software ID in the Return Header must have passed testing for the form family and ‘TaxYr’.";
                errorList1.add(ruleNum);
                errorList1.add(severityCd);
                errorList1.add(errorMessageTxt);
                validationErrorMap.get(submissionId).add(errorList1);
                ruleNum = "F1040-525-03";
                severityCd = "Reject and Stop";
                errorMessageTxt =
                        "If 'PINTypeCd' in the Return Header has the value \"Self-Select On-Line\" and the filing status of the return is married filing jointly and Form 1040, [ 'SpecialProcessingLiteralCd' and 'CombatZoneCd' and 'SpecialProcessingCodeTxt' and 'PrimaryDeathDt' ] do not have values, then 'PrimaryBirthDt' in the Return Header must match the e-File database.";
                errorList2.add(ruleNum);
                errorList2.add(severityCd);
                errorList2.add(errorMessageTxt);
                validationErrorMap.get(submissionId).add(errorList2);
            }
        });
        Iterable<Completed> completeds = acknowledgementService.createNewCompleteds(statusSubmissionIdMap);
        return new ValidationMapCompleted(validationErrorMap, completeds);
    }

    @Test
    void stripPendingAcknowledgementsFromStatusSubmissionIdMap() {
        Map<String, List<String>> statusSubmissionIdMap = getStatusSubmissionIdMap();
        Map<String, List<String>> newMap =
                acknowledgementService.stripPendingAcknowledgementsFromStatusSubmissionIdMap(statusSubmissionIdMap);
        assertNull(newMap.get("pending"));
        assertEquals(newMap.get("rejected").size(), 1);
        assertEquals(newMap.get("accepted").size(), 1);
    }

    @Test
    void addToSubmissionIdMapErrorDoesNotMutateSubmissionIdMap() {
        Map<String, List<String>> statusSubmissionIdMap = getStatusSubmissionIdMap();
        String status = "error";
        String submissionId = UUID.randomUUID().toString().substring(0, 20);

        acknowledgementService.addToSubmissionIdMap(status, submissionId, statusSubmissionIdMap);
        assertNull(statusSubmissionIdMap.get("error"));
        assertEquals(statusSubmissionIdMap.get("rejected").size(), 1);
        assertEquals(statusSubmissionIdMap.get("accepted").size(), 1);
        assertEquals(statusSubmissionIdMap.get("pending").size(), 1);
    }

    @Test
    void createNewCompletedsCreatesExpectedEntities() {
        assertEquals(acknowledgementService.getAllPending().spliterator().getExactSizeIfKnown(), 0);
        assertEquals(acknowledgementService.getAllCompleted().spliterator().getExactSizeIfKnown(), 0);
        Map<String, List<String>> statusSubmssionIdMap = getStatusSubmissionIdMap();
        Iterable<Completed> completeds = acknowledgementService.createNewCompleteds(statusSubmssionIdMap);
        completedRepo.saveAll(completeds);
        // one pending is created, two completed (rejected and accepted)
        assertEquals(acknowledgementService.getAllPending().spliterator().getExactSizeIfKnown(), 1);
        assertEquals(acknowledgementService.getAllCompleted().spliterator().getExactSizeIfKnown(), 2);
    }

    @Test
    void bulkUpdateEntitiesWithNewErrors() {
        assertEquals(acknowledgementService.getAllPending().spliterator().getExactSizeIfKnown(), 0);
        assertEquals(acknowledgementService.getAllError().spliterator().getExactSizeIfKnown(), 0);
        assertEquals(acknowledgementService.getAllCompleted().spliterator().getExactSizeIfKnown(), 0);

        ValidationMapCompleted validationMapCompleted = getValidationErrorMap();
        Map<String, List<List<String>>> validationErrorMap = validationMapCompleted.validationErrorMap;
        Iterable<Completed> c = validationMapCompleted.completeds;
        Map<String, Completed> cmap = new HashMap<>();
        Map<String, Pending> pmap = new HashMap<>();
        c.forEach(completed -> cmap.put(completed.getSubmissionId(), completed));

        String podId = getDefaultPodIdentifierFromProperties(0, "us-gov-east-1");
        Iterable<Pending> p = pendingRepo.findAllByPodId(podId);
        p.forEach(pending -> pmap.put(pending.getSubmissionId(), pending));

        acknowledgementService.bulkUpdateEntities(validationErrorMap, cmap, pmap);

        String rejectedSubmissionId =
                validationErrorMap.entrySet().iterator().next().getKey();
        Completed rejectedAck = acknowledgementService
                .getCompletedBySubmissionId(rejectedSubmissionId)
                .get();

        Pending pending = acknowledgementService.getAllPending().iterator().next();
        Iterator<Completed> completeds =
                acknowledgementService.getAllCompleted().iterator();

        assertEquals(acknowledgementService.getAllCompleted().spliterator().getExactSizeIfKnown(), 2);
        assertEquals(acknowledgementService.getAllError().spliterator().getExactSizeIfKnown(), 2);
        assertEquals(rejectedAck.getSubmissionId(), rejectedSubmissionId);
        assertNotNull(rejectedAck.getErrors());

        // the pending object that corresponded to the rejected completed is deleted, the one that remains
        // is still pending from the standpoint of MeF so it was not deleted
        assertEquals(acknowledgementService.getAllPending().spliterator().getExactSizeIfKnown(), 1);
        completeds.forEachRemaining(completed -> {
            assertNotEquals(pending.getSubmissionId(), completed.getSubmissionId());
        });
    }

    @Test
    void bulkUpdateEntitiesWithPreexistingErrors() {
        ValidationMapCompleted validationMapCompleted = getValidationErrorMap();
        Map<String, List<List<String>>> validationErrorMap1 = validationMapCompleted.validationErrorMap;
        Iterable<Completed> c = validationMapCompleted.completeds;
        Map<String, Completed> cmap = new HashMap<>();
        Map<String, Pending> pmap = new HashMap<>();
        c.forEach(completed -> cmap.put(completed.getSubmissionId(), completed));

        String podId = getDefaultPodIdentifierFromProperties(0, "us-gov-east-1");
        Iterable<Pending> p = pendingRepo.findAllByPodId(podId);
        p.forEach(pending -> pmap.put(pending.getSubmissionId(), pending));

        acknowledgementService.bulkUpdateEntities(validationErrorMap1, cmap, pmap);

        String rejectedSubmissionId =
                validationErrorMap1.entrySet().iterator().next().getKey();
        Completed rejectedAck = acknowledgementService
                .getCompletedBySubmissionId(rejectedSubmissionId)
                .get();
        Pending pending = acknowledgementService.getAllPending().iterator().next();
        Iterator<Completed> completeds =
                acknowledgementService.getAllCompleted().iterator();

        assertEquals(acknowledgementService.getAllCompleted().spliterator().getExactSizeIfKnown(), 2);
        assertEquals(acknowledgementService.getAllError().spliterator().getExactSizeIfKnown(), 2);
        assertEquals(rejectedAck.getSubmissionId(), rejectedSubmissionId);
        assertNotNull(rejectedAck.getErrors());
        assertEquals(acknowledgementService.getAllPending().spliterator().getExactSizeIfKnown(), 1);
        completeds.forEachRemaining(completed -> {
            assertNotEquals(pending.getSubmissionId(), completed.getSubmissionId());
        });

        // mimic polling MeF again for the same errors but different submissions
        // we would expect the rejected ack in the second batch to still have associated errors
        ValidationMapCompleted validationMapCompleted2 = getValidationErrorMap();
        Map<String, List<List<String>>> validationErrorMap2 = validationMapCompleted2.validationErrorMap;
        Iterable<Completed> c2 = validationMapCompleted2.completeds;
        Map<String, Completed> cmap2 = new HashMap<>();
        c2.forEach(completed -> cmap2.put(completed.getSubmissionId(), completed));

        acknowledgementService.bulkUpdateEntities(validationErrorMap2, cmap2, pmap);
        String rejectedSubmissionId2 =
                validationErrorMap2.entrySet().iterator().next().getKey();
        Completed rejectedAck2 = acknowledgementService
                .getCompletedBySubmissionId(rejectedSubmissionId2)
                .get();
        Pending updatedPendings =
                acknowledgementService.getAllPending().iterator().next();
        Iterator<Completed> updatedCompleteds =
                acknowledgementService.getAllCompleted().iterator();

        // added another rejected and accepted submission object on top of the 2 in the first batch
        assertEquals(acknowledgementService.getAllCompleted().spliterator().getExactSizeIfKnown(), 4);
        // error count is the same because each error in the db is unique
        assertEquals(acknowledgementService.getAllError().spliterator().getExactSizeIfKnown(), 2);

        assertEquals(rejectedAck2.getSubmissionId(), rejectedSubmissionId2);
        // errors are still persisted
        assertNotNull(rejectedAck2.getErrors());
        assertEquals(acknowledgementService.getAllPending().spliterator().getExactSizeIfKnown(), 2);
        updatedCompleteds.forEachRemaining(completed -> {
            assertNotEquals(updatedPendings.getSubmissionId(), completed.getSubmissionId());
        });
        assertEquals(
                rejectedAck2.getErrors().getFirst().getErrorMessage(),
                rejectedAck.getErrors().getFirst().getErrorMessage());
        assertEquals(
                rejectedAck2.getErrors().getLast().getErrorMessage(),
                rejectedAck.getErrors().getLast().getErrorMessage());
    }

    @Test
    void getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission_returnsAcceptedSubmissionEvenIfNotTheLatest() {
        UUID taxReturnId = UUID.randomUUID();

        String olderAcceptedSubmissionId = "accepted";
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, olderAcceptedSubmissionId));

        String latestSubmissionId = "latest";
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, latestSubmissionId));

        Completed acceptedCompleted = new Completed();
        acceptedCompleted.setSubmissionId(olderAcceptedSubmissionId);
        acceptedCompleted.setStatus("accepted");
        completedRepo.save(acceptedCompleted);

        Completed latestCompleted = new Completed();
        latestCompleted.setSubmissionId(latestSubmissionId);
        latestCompleted.setStatus("rejected");
        completedRepo.save(latestCompleted);

        var submissionId =
                acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(taxReturnId);
        assertEquals(olderAcceptedSubmissionId, submissionId);
    }

    @Test
    void
            getLatestAcceptedSubmissionIdOfParentTaxReturn_getsTheLatestAcceptedSubmissionIdOfTheParentTaxReturnEvenIfNotTheLatest() {
        UUID taxReturnId = UUID.randomUUID();

        String olderAcceptedSubmissionId = "accepted";
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, olderAcceptedSubmissionId));

        String latestSubmissionId = "latest";
        taxReturnSubmissionRepository.save(new TaxReturnSubmission(taxReturnId, latestSubmissionId));

        Completed acceptedCompleted = new Completed();
        acceptedCompleted.setSubmissionId(olderAcceptedSubmissionId);
        acceptedCompleted.setStatus("accepted");
        completedRepo.save(acceptedCompleted);

        Completed latestCompleted = new Completed();
        latestCompleted.setSubmissionId(latestSubmissionId);
        latestCompleted.setStatus("rejected");
        completedRepo.save(latestCompleted);

        var submissionId =
                acknowledgementService.getLatestAcceptedSubmissionIdOfParentTaxReturn(olderAcceptedSubmissionId);
        assertTrue(submissionId.isPresent());
        assertEquals(olderAcceptedSubmissionId, submissionId.get());
    }

    @Test
    void getRejectionCodesForSubmissionId_returnsRejectionCodes() {
        Error error1 = new Error();
        error1.setMefErrorCode("code1");
        error1.setErrorCodeTranslationKey("key1");
        error1.setMefErrorCategory("category1");
        error1.setErrorMessage("message1");
        error1 = errorRepository.save(error1);

        Completed completedRejection = new Completed();
        String submissionId = "rejected";
        completedRejection.setSubmissionId(submissionId);
        completedRejection.setStatus("rejected");
        completedRejection.setErrors(List.of(error1));
        completedRepo.save(completedRejection);

        List<RejectedStatus> rejectionCodes = acknowledgementService.getRejectionCodesForSubmissionId(submissionId);

        assertEquals(1, rejectionCodes.size());
        RejectedStatus rejectedStatus = rejectionCodes.getFirst();
        assertEquals(error1.getMefErrorCode(), rejectedStatus.MeFErrorCode);
        assertEquals(error1.getErrorCodeTranslationKey(), rejectedStatus.TranslationKey);
        assertEquals(error1.getErrorMessage(), rejectedStatus.MeFDescription);
    }

    @Test
    void getRejectionCodesForSubmissionId_returnsEmptyListWhenCompletedIsNotRejected() {
        Completed completedAccepted = new Completed();
        String submissionId = "accepted";
        completedAccepted.setSubmissionId(submissionId);
        completedAccepted.setStatus("accepted");
        completedRepo.save(completedAccepted);

        List<RejectedStatus> rejectionCodes = acknowledgementService.getRejectionCodesForSubmissionId(submissionId);

        assertEquals(0, rejectionCodes.size());
    }

    @Test
    void getRejectionCodesForSubmissionId_throwsExceptionWhenNoCompletedRecord() {
        String submissionId = "missing";
        EntityNotFoundException thrown = assertThrows(
                EntityNotFoundException.class,
                () -> acknowledgementService.getRejectionCodesForSubmissionId(submissionId));

        assertEquals("Could not find completed record for submission ID: " + submissionId, thrown.getMessage());
    }

    @Test
    void whenCreateServiceContextWrapper_WithPodIdentifiers_ThenReturnsAsidFromDatabase() {
        PodIdentifier p = createPodIdentifer("us-gov-east-1", statusProperties.getAsid(), 0);
        podIdentifierRepository.save(p);

        String asid = podIdentifierRepository.findAsidByPodId(p.getPodId()).get();

        ServiceContextWrapper serviceContextWrapper = acknowledgementService.createServiceContextWrapper();
        ServiceContext serviceContext = serviceContextWrapper.getServiceContext();
        assertNotNull(serviceContextWrapper);
        assertNotNull(serviceContextWrapper.getServiceContext());
        assertEquals(serviceContext.getAppSysID(), statusProperties.getAsid());
        assertEquals(serviceContext.getAppSysID(), asid);
        assertEquals(serviceContext.getEtin().toString(), statusProperties.getEtin());
        assertEquals(serviceContext.getTestCdType(), TestCdType.T);
    }

    @Test
    void whenCreateServiceContextWrapper_WithMultiplePodIdentifiers_ThenReturnsCorrectAsid() {
        PodIdentifier p1 = createPodIdentifer("us-gov-east-1", ASID, 2);
        PodIdentifier p2 = createPodIdentifer("us-gov-east-1", ASID + "2", 1);
        PodIdentifier p3 = createPodIdentifer("us-gov-east-1", statusProperties.getAsid(), 0);
        podIdentifierRepository.save(p1);
        podIdentifierRepository.save(p2);
        podIdentifierRepository.save(p3);

        String correctAsid =
                podIdentifierRepository.findAsidByPodId(p3.getPodId()).get();

        ServiceContextWrapper serviceContextWrapper = acknowledgementService.createServiceContextWrapper();
        ServiceContext serviceContext = serviceContextWrapper.getServiceContext();
        assertNotNull(serviceContextWrapper);
        assertNotNull(serviceContextWrapper.getServiceContext());
        assertEquals(serviceContext.getAppSysID(), statusProperties.getAsid());
        assertEquals(serviceContext.getAppSysID(), correctAsid);
        assertEquals(serviceContext.getEtin().toString(), statusProperties.getEtin());
        assertEquals(serviceContext.getTestCdType(), TestCdType.T);
    }

    private record ValidationMapCompleted(
            Map<String, List<List<String>>> validationErrorMap, Iterable<Completed> completeds) {}

    @Test
    void whenRejectedStatusChangeMessages_DoPublish() {
        List<AcknowledgementWrapper> acksList = new ArrayList<>();
        PodIdentifier pi = createPodIdentifer("us-gov-east-1", statusProperties.getAsid(), 0);
        podIdentifierRepository.save(pi);
        Pending pending = new Pending("submissionId", pi.getPodId());
        pendingRepo.save(pending);
        acksList.add(new AcknowledgementWrapper("submissionId", "reeee", "rejected"));
        GetAcksResultWrapper acksResult = new GetAcksResultWrapper(new AcknowledgementsListWrapper(acksList));
        acknowledgementService.bulkUpdateRecordsFromAckResultAndEnqueueStatusChangeMessages(
                acksResult, List.of(pending));
        Map<String, List<String>> statusSubmissionIdMap = new HashMap<>();
        statusSubmissionIdMap.put("accepted", List.of());
        statusSubmissionIdMap.put("rejected", List.of("submissionId"));
        verify(statusChangeMessageService, times(1)).publishStatusChangePayloadV1(statusSubmissionIdMap);
    }

    @Test
    void whenAcceptedStatusChangeMessages_DoPublish() {
        List<AcknowledgementWrapper> acksList = new ArrayList<>();
        PodIdentifier pi = createPodIdentifer("us-gov-east-1", statusProperties.getAsid(), 0);
        podIdentifierRepository.save(pi);
        Pending pending = new Pending("submissionId3", pi.getPodId());
        pendingRepo.save(pending);
        acksList.add(new AcknowledgementWrapper("submissionId3", "beeee", "accepted"));
        GetAcksResultWrapper acksResult = new GetAcksResultWrapper(new AcknowledgementsListWrapper(acksList));
        acknowledgementService.bulkUpdateRecordsFromAckResultAndEnqueueStatusChangeMessages(
                acksResult, List.of(pending));
        Map<String, List<String>> statusSubmissionIdMap = new HashMap<>();
        statusSubmissionIdMap.put("accepted", List.of("submissionId3"));
        statusSubmissionIdMap.put("rejected", List.of());
        verify(statusChangeMessageService, times(1)).publishStatusChangePayloadV1(statusSubmissionIdMap);
    }

    @Test
    void whenAcceptedandRejectedStatusChangeMessages_DoPublish() {
        List<AcknowledgementWrapper> acksList = new ArrayList<>();
        PodIdentifier pi = createPodIdentifer("us-gov-east-1", statusProperties.getAsid(), 0);
        podIdentifierRepository.save(pi);
        Pending pending = new Pending("subId3", pi.getPodId());
        Pending pending2 = new Pending("subId", pi.getPodId());
        pendingRepo.save(pending);
        pendingRepo.save(pending2);
        acksList.add(new AcknowledgementWrapper("subId3", "yerp", "accepted"));
        acksList.add(new AcknowledgementWrapper("subId", "nerp", "rejected"));
        GetAcksResultWrapper acksResult = new GetAcksResultWrapper(new AcknowledgementsListWrapper(acksList));
        acknowledgementService.bulkUpdateRecordsFromAckResultAndEnqueueStatusChangeMessages(
                acksResult, List.of(pending, pending2));
        Map<String, List<String>> statusSubmissionIdMap = new HashMap<>();
        statusSubmissionIdMap.put("accepted", List.of("subId3"));
        statusSubmissionIdMap.put("rejected", List.of("subId"));
        verify(statusChangeMessageService, times(1)).publishStatusChangePayloadV1(statusSubmissionIdMap);
    }

    @Test
    void whenNoStatusChangeMessages_NoPublish() {
        List<AcknowledgementWrapper> acksList = new ArrayList<>();
        GetAcksResultWrapper acksResult = new GetAcksResultWrapper(new AcknowledgementsListWrapper(acksList));
        acknowledgementService.bulkUpdateRecordsFromAckResultAndEnqueueStatusChangeMessages(
                acksResult, new ArrayList<>());
        Map<String, List<String>> statusSubmissionIdMap = new HashMap<>();
        statusSubmissionIdMap.put("accepted", List.of());
        statusSubmissionIdMap.put("rejected", List.of());
        verify(statusChangeMessageService, times(0)).publishStatusChangePayloadV1(statusSubmissionIdMap);
    }
}
