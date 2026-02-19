package gov.irs.directfile.status.error;

import ch.qos.logback.classic.Level;
import jakarta.validation.ConstraintViolationException;
import org.hibernate.engine.jdbc.batch.JdbcBatchLogging;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.transaction.TransactionSystemException;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import gov.irs.directfile.status.domain.ToolkitError;
import gov.irs.directfile.status.extension.BatchUtil;
import gov.irs.directfile.status.extension.LoggerExtension;

import static org.junit.jupiter.api.Assertions.*;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ImportAutoConfiguration(classes = SecurityAutoConfiguration.class)
@Transactional(propagation = Propagation.NOT_SUPPORTED) // Allows CrudRepository Transactional scope to work as intended
@DataJpaTest
public class ToolkitErrorRepositoryTest {
    @Autowired
    ToolkitErrorRepository toolkitErrorRepo;

    @RegisterExtension
    private static final LoggerExtension batchLogVerifier = new LoggerExtension(Level.TRACE, JdbcBatchLogging.NAME);

    @Value("${spring.jpa.properties.hibernate.jdbc.batch_size}")
    private int batchSize;

    @Test
    public void DoesNotAllowNullErrorName() {
        ToolkitError tke = new ToolkitError();
        tke.setSubmissionId("12345620230215000010");
        tke.setErrorMessage("something");
        TransactionSystemException tse = assertThrows(TransactionSystemException.class, () -> {
            toolkitErrorRepo.save(tke);
        });

        assertEquals(
                ConstraintViolationException.class, tse.getMostSpecificCause().getClass());
    }

    @Test
    public void DoesNotAllowNullErrorMessage() {
        ToolkitError tke = new ToolkitError();
        tke.setSubmissionId("12345678900987654321");
        tke.setErrorName("ToolkitException");
        TransactionSystemException tse = assertThrows(TransactionSystemException.class, () -> {
            toolkitErrorRepo.save(tke);
        });
        assertEquals(
                ConstraintViolationException.class, tse.getMostSpecificCause().getClass());
    }

    @Test
    public void SavesWithValidFields() {
        ToolkitError tke = new ToolkitError();
        tke.setSubmissionId("12345678900987654321");
        tke.setErrorName("ToolkitException");
        tke.setErrorMessage("Something went wrong!");

        toolkitErrorRepo.save(tke);

        assert (toolkitErrorRepo.findById(tke.getSubmissionId())).isPresent();

        // Verify JDBC batching
        batchLogVerifier.verifyLogContainsMessage(
                BatchUtil.buildBatchMessage(1, batchSize, ToolkitError.class.getName(), BatchUtil.BatchType.INSERT));
    }
}
