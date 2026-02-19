package gov.irs.directfile.status.acknowledgement;

import java.util.List;

import ch.qos.logback.classic.Level;
import org.hibernate.engine.jdbc.batch.JdbcBatchLogging;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import gov.irs.directfile.status.domain.Completed;
import gov.irs.directfile.status.domain.Error;
import gov.irs.directfile.status.error.ErrorRepository;
import gov.irs.directfile.status.extension.BatchUtil;
import gov.irs.directfile.status.extension.LoggerExtension;

import static org.junit.jupiter.api.Assertions.*;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ImportAutoConfiguration(classes = SecurityAutoConfiguration.class)
@DataJpaTest(properties = "spring.main.web-application-type=servlet")
class CompletedAcknowledgementRepositoryTest {

    @Autowired
    CompletedAcknowledgementRepository completedRepo;

    @Autowired
    ErrorRepository errorRepository;

    @RegisterExtension
    private static final LoggerExtension batchLogVerifier = new LoggerExtension(Level.TRACE, JdbcBatchLogging.NAME);

    @Value("${spring.jpa.properties.hibernate.jdbc.batch_size}")
    private int batchSize;

    @Test
    public void CanAddAnAccepted() {
        Completed c = new Completed();
        c.setSubmissionId("12345620230215000001");
        c.setStatus("Accepted");
        completedRepo.save(c);
        var completed =
                completedRepo.GetCompletedSubmission("12345620230215000001").get();
        assertEquals("Accepted", completed.getStatus());

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Completed.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanAddARejected() {
        Completed c = new Completed();
        c.setSubmissionId("12345620230215000001");
        c.setStatus("Rejected");
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e.setMefErrorCategory("Reject and Stop");
        c.setErrors(List.of(e));
        // must add error to database first!
        errorRepository.save(e);
        completedRepo.save(c);
        var completed =
                completedRepo.GetCompletedSubmission("12345620230215000001").get();
        assertEquals("Rejected", completed.getStatus());
        assertEquals("XML-123-4567-006", completed.getErrors().get(0).getMefErrorCode());

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Error.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Completed.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(BatchUtil.buildBatchMessage(
                1, batchSize, Completed.class.getName() + ".errors", BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanAddARejectedWithMultipleErrors() {
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

        var completed =
                completedRepo.GetCompletedSubmission("12345620230215000001").get();
        assertEquals("Rejected", completed.getStatus());
        assertEquals("XML-123-4567-006", completed.getErrors().get(0).getMefErrorCode());
        assertEquals("REJC-00001", completed.getErrors().get(1).getMefErrorCode());

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(2, batchSize, Error.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Completed.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(BatchUtil.buildBatchMessage(
                2, batchSize, Completed.class.getName() + ".errors", BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanQueryForANonExistentSubmissionId() {
        var completed = completedRepo.GetCompletedSubmission("12345620230215000001");
        assertTrue(completed.isEmpty());
    }

    @Test
    public void CanAddTheSameErrorToTwoDifferentCompletedRecords() {
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e.setMefErrorCategory("Reject and Stop");
        errorRepository.save(e);

        Completed c = new Completed();
        c.setSubmissionId("12345620230215000001");
        c.setStatus("Rejected");
        c.setErrors(List.of(e));
        completedRepo.save(c);

        Completed c2 = new Completed();
        c2.setSubmissionId("12345620230215000002");
        c2.setStatus("Rejected");
        c2.setErrors(List.of(e));
        completedRepo.save(c2);

        assertEquals(1, errorRepository.findAll().spliterator().getExactSizeIfKnown());
        assertEquals(2, completedRepo.findAll().spliterator().getExactSizeIfKnown());

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Error.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(2, batchSize, Completed.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(BatchUtil.buildBatchMessage(
                2, batchSize, Completed.class.getName() + ".errors", BatchUtil.BatchType.INSERT));
    }
}
