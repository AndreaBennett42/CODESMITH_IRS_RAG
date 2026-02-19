package gov.irs.directfile.api.util.base;

import org.junit.jupiter.api.AfterEach;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.context.SpringBootTest;

import gov.irs.directfile.api.authentication.UserDetailsCacheService;
import gov.irs.directfile.api.user.UserRepository;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
public abstract class BaseRealSecurityFilterTest {
    @Autowired
    UserRepository userRepository;

    @Autowired
    UserDetailsCacheService userDetailsCacheService;

    @AfterEach
    public void reset() {
        userRepository.deleteAll();
        userDetailsCacheService.clear();
    }
}
