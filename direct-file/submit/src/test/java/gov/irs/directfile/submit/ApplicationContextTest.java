package gov.irs.directfile.submit;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;

import gov.irs.directfile.submit.config.SnsClientTestConfiguration;
import gov.irs.directfile.submit.config.SynchronousS3TestConfiguration;
import gov.irs.directfile.submit.service.SqsConnectionSetupService;

@SpringBootTest
@Import({SynchronousS3TestConfiguration.class, SnsClientTestConfiguration.class})
public class ApplicationContextTest {

    @MockBean
    SqsConnectionSetupService sqsConnectionSetupService;

    /**
     * This empty test validates that the Spring Application
     * will successfully start up.
     *
     * The @SpringBootTest loads in the application context
     * for any test in the file. So this empty test is enough to validate
     * if the app will start up.
     *
     *
     * */
    @Test
    public void contextLoads() {}
}
