package gov.irs.directfile.api.util.base;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.test.web.servlet.MockMvc;

import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.user.UserRepository;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.api.util.SecurityTestConfiguration;
import gov.irs.directfile.api.util.TestDataFactory;

@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
public abstract class BaseIntegrationTest extends BaseControllerTest {
    @Autowired
    public MockMvc mvc;

    @Autowired
    public TaxReturnRepository taxReturnRepository;

    @Autowired
    public UserRepository userRepository;

    @Autowired
    public TestDataFactory testDataFactory;

    @BeforeEach
    void createUsers() {
        for (SecurityTestConfiguration.TestUserProperties testUserProperties :
                SecurityTestConfiguration.testUserMap.values()) {
            User user = testDataFactory.createUserFromTestUser(testUserProperties);
            testUserProperties.setInternalId(user.getId());
        }
    }

    @AfterEach
    public void resetDb() {
        userRepository.deleteAll();
        taxReturnRepository.deleteAll();
    }
}
