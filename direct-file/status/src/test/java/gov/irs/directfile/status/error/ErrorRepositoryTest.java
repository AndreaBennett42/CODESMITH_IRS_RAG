package gov.irs.directfile.status.error;

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
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

import gov.irs.directfile.status.acknowledgement.CompletedAcknowledgementRepository;
import gov.irs.directfile.status.domain.Completed;
import gov.irs.directfile.status.domain.Error;
import gov.irs.directfile.status.extension.BatchUtil;
import gov.irs.directfile.status.extension.LoggerExtension;

import static org.junit.jupiter.api.Assertions.*;

// For right now, Errors only exist as information on
// the Reject.  It doesn't need a direct access right now.
// It may/will in the future.
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ImportAutoConfiguration(classes = SecurityAutoConfiguration.class)
@DataJpaTest(properties = "spring.main.web-application-type=servlet")
class ErrorRepositoryTest {

    @Autowired
    CompletedAcknowledgementRepository completedRepo;

    @Autowired
    ErrorRepository errorRepository;

    @Autowired
    private TestEntityManager testEntityManager;

    @RegisterExtension
    private static final LoggerExtension batchLogVerifier = new LoggerExtension(Level.TRACE, JdbcBatchLogging.NAME);

    @Value("${spring.jpa.properties.hibernate.jdbc.batch_size}")
    private int batchSize;

    @Test
    public void CanAddABasicError() {
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e.setMefErrorCategory("Reject");
        errorRepository.save(e);
        Error retrieved = errorRepository.findById("XML-123-4567-006").get();
        assertEquals(e.getErrorCodeTranslationKey(), retrieved.getErrorCodeTranslationKey());
        assertEquals(e.getMefErrorCode(), retrieved.getMefErrorCode());
        assertEquals(e.getErrorMessage(), retrieved.getErrorMessage());
        assertEquals(e.getMefErrorCategory(), retrieved.getMefErrorCategory());

        // Verify JDBC batching
        testEntityManager.flush();
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Error.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanAddAnErrorToACompletedRecord() {
        // you should never do it this way!
        // Errors should be created first
        Completed c = new Completed();
        c.setSubmissionId("12345620230215000001");
        c.setStatus("Rejected");
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e.setMefErrorCategory("Reject");
        e.setCompleted(List.of(c));
        completedRepo.save(c);
        errorRepository.save(e);
        Error retrieved = errorRepository.findById("XML-123-4567-006").get();
        assertEquals("12345620230215000001", retrieved.getCompleted().get(0).getSubmissionId());

        // Verify JDBC batching
        testEntityManager.flush();
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Completed.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Error.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CallingSaveOnAKnownIDWithNewInformationUpdatesTheRecord() {
        Error e = new Error();
        e.setMefErrorCode("XML-123-4567-006");
        e.setErrorMessage("You messed up!");
        e.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e.setMefErrorCategory("Reject");
        errorRepository.save(e);
        Error e2 = new Error();
        e2.setMefErrorCode("XML-123-4567-006");
        e2.setErrorMessage("This is bad");
        e2.setErrorCodeTranslationKey("translation/reject/XML-123-4567-006");
        e2.setMefErrorCategory("Reject");
        errorRepository.save(e2);

        Error retrieved = errorRepository.findById("XML-123-4567-006").get();
        assertEquals(e.getErrorCodeTranslationKey(), retrieved.getErrorCodeTranslationKey());
        assertEquals(e.getMefErrorCode(), retrieved.getMefErrorCode());
        assertEquals(e2.getErrorMessage(), retrieved.getErrorMessage());
        assertEquals(e.getMefErrorCategory(), retrieved.getMefErrorCategory());

        // Verify JDBC batching
        testEntityManager.flush();
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Error.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Error.class.getName(), BatchUtil.BatchType.UPDATE));
    }
}
