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
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

import gov.irs.directfile.status.domain.Pending;
import gov.irs.directfile.status.extension.BatchUtil;
import gov.irs.directfile.status.extension.LoggerExtension;

import static org.junit.jupiter.api.Assertions.*;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ImportAutoConfiguration(classes = SecurityAutoConfiguration.class)
@DataJpaTest(properties = "spring.main.web-application-type=servlet")
public class PendingAcknowledgementRepositoryTest {
    @Autowired
    PendingAcknowledgementRepository pendingRepo;

    @Autowired
    private TestEntityManager testEntityManager;

    @RegisterExtension
    private static final LoggerExtension batchLogVerifier = new LoggerExtension(Level.TRACE, JdbcBatchLogging.NAME);

    @Value("${spring.jpa.properties.hibernate.jdbc.batch_size}")
    private int batchSize;

    @Test
    public void CanAddASubmissionIdToTheTable() {
        Pending p = new Pending();
        p.setSubmissionId("12345620230215000010");
        pendingRepo.save(p);

        // Verify JDBC batching
        testEntityManager.flush();
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Pending.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanAddMultipleSubmissionsToTheTable() {
        Pending p = new Pending();
        p.setSubmissionId("12345620230215000001");
        pendingRepo.save(p);
        Pending p2 = new Pending();
        p2.setSubmissionId("12345620230215000002");
        pendingRepo.save(p2);
        Pending p3 = new Pending();
        Pending p4 = new Pending();
        p3.setSubmissionId("12345620230215000003");
        p4.setSubmissionId("12345620230215000004");
        var multiple = List.of(p3, p4);
        pendingRepo.saveAll(multiple);

        // Verify JDBC batching
        testEntityManager.flush();
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(4, batchSize, Pending.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanRetrieveFromPendingTable() {
        Pending p = new Pending();
        p.setSubmissionId("12345620230215000001");
        pendingRepo.save(p);
        var pending = pendingRepo.GetPendingSubmission("12345620230215000001");
        assertEquals(p.getSubmissionId(), pending.get().getSubmissionId());

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Pending.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanDeleteFromPendingTable() {
        Pending p = new Pending();
        p.setSubmissionId("12345620230215000001");
        pendingRepo.save(p);
        Pending p2 = new Pending();
        p2.setSubmissionId("12345620230215000001");
        pendingRepo.delete(p2);
        var option = pendingRepo.GetPendingSubmission("12345620230215000001");
        assertTrue(option.isEmpty());

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Pending.class.getName(), BatchUtil.BatchType.INSERT));
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Pending.class.getName(), BatchUtil.BatchType.DELETE));
    }

    @Test
    public void CannotEnterTwoOfTheSameSubmissionIdsIntoThePendingTable() {
        Pending p = new Pending();
        p.setSubmissionId("12345620230215000001");
        pendingRepo.save(p);
        Pending p2 = new Pending();
        p2.setSubmissionId("12345620230215000001");
        pendingRepo.save(p2);
        var all = pendingRepo.findAll();
        var size = all.spliterator().getExactSizeIfKnown();
        assertEquals(1, size);

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Pending.class.getName(), BatchUtil.BatchType.INSERT));
    }

    @Test
    public void CanQueryForANonExistentPendingWithNoResults() {
        var empty = pendingRepo.GetPendingSubmission("12345620230215000001");
        assertTrue(empty.isEmpty());
    }

    @Test
    public void CannotEnterASubmissionIdThatIsTooLong() {
        Pending p = new Pending();
        p.setSubmissionId("123456202302150000001");
        assertEquals(0, pendingRepo.findAll().spliterator().getExactSizeIfKnown());
        // I wish I knew why this saves
        // TODO: make this throw...
        pendingRepo.save(p);

        assertThrows(org.springframework.dao.DataIntegrityViolationException.class, () -> {
            assertEquals(0, pendingRepo.findAll().spliterator().getExactSizeIfKnown());
        });

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, Pending.class.getName(), BatchUtil.BatchType.INSERT));
    }
}
