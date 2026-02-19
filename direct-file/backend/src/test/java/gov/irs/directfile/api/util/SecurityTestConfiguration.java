package gov.irs.directfile.api.util;

import java.util.Map;
import java.util.UUID;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.web.SecurityFilterChain;

import gov.irs.directfile.api.authentication.NullAuthenticationException;
import gov.irs.directfile.api.config.identity.IdentityAttributes;
import gov.irs.directfile.api.config.identity.IdentitySupplier;
import gov.irs.directfile.api.user.UserRepository;
import gov.irs.directfile.api.user.domain.*;

@TestConfiguration
@Slf4j
public class SecurityTestConfiguration {

    private final UserRepository userRepository;

    @Value("${direct-file.api-version}")
    private String api_v;

    private static final String STATE_EXPORTED_FACTS = "/state-api/state-exported-facts/**";
    private static final String STATE_TAX_RETURN_STATUS = "/state-api/status/**";

    public SecurityTestConfiguration(final UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        log.info("Running with test security configuration");
        return http.build();
    }

    public static final String TEST_USER_1 = "11111111-bb2e-498a-b102-c8fdad502ba3";
    public static final UUID TEST_USER_1_INTERNAL_ID = UUID.fromString(TEST_USER_1);
    public static final UUID TEST_USER_1_EXTERNAL_ID = UUID.fromString("11111111-1111-1111-1111-111111111111");
    public static final String TEST_USER_1_EMAIL = "test.user.1@direct-file.local";
    public static final String TEST_USER_1_TIN = "123001111";
    public static final UserInfo TEST_USER_INFO_1 =
            new UserInfo(TEST_USER_1_INTERNAL_ID, TEST_USER_1_EXTERNAL_ID, TEST_USER_1_EMAIL, TEST_USER_1_TIN);
    public static final String TEST_USER_2 = "22222222-6bb4-46ef-8f25-f341b5ff335c";
    public static final UUID TEST_USER_2_INTERNAL_ID = UUID.fromString(TEST_USER_2);
    public static final UUID TEST_USER_2_EXTERNAL_ID = UUID.fromString("22222222-2222-2222-2222-222222222222");
    public static final String TEST_USER_2_EMAIL = "test.user.2@direct-file.local";
    public static final String TEST_USER_2_TIN = "123002222";
    public static final UserInfo TEST_USER_INFO_2 =
            new UserInfo(TEST_USER_2_INTERNAL_ID, TEST_USER_2_EXTERNAL_ID, TEST_USER_2_EMAIL, TEST_USER_2_TIN);
    public static final String TEST_USER_3 = "33333333-9634-49b3-860a-05452b6478f1";
    public static final UUID TEST_USER_3_INTERNAL_ID = UUID.fromString(TEST_USER_3);
    public static final UUID TEST_USER_3_EXTERNAL_ID = UUID.fromString("33333333-3333-3333-3333-333333333333");
    public static final String TEST_USER_3_EMAIL = "test.user.3@direct-file.local";
    public static final String TEST_USER_3_TIN = "123003333";
    public static final UserInfo TEST_USER_INFO_3 =
            new UserInfo(TEST_USER_3_INTERNAL_ID, TEST_USER_3_EXTERNAL_ID, TEST_USER_3_EMAIL, TEST_USER_3_TIN);

    @Getter
    public static class TestUserProperties {
        @Setter
        UUID internalId;

        UUID externalId;
        String email;
        String tin;

        public TestUserProperties(UUID externalId, String email, String tin) {
            this.externalId = externalId;
            this.email = email;
            this.tin = tin;
        }
    }

    public static final Map<String, TestUserProperties> testUserMap = Map.of(
            TEST_USER_1, new TestUserProperties(TEST_USER_1_EXTERNAL_ID, TEST_USER_1_EMAIL, TEST_USER_1_TIN),
            TEST_USER_2, new TestUserProperties(TEST_USER_2_EXTERNAL_ID, TEST_USER_2_EMAIL, TEST_USER_2_TIN),
            TEST_USER_3, new TestUserProperties(TEST_USER_3_EXTERNAL_ID, TEST_USER_3_EMAIL, TEST_USER_3_TIN));

    @Bean
    public IdentitySupplier getTestIdentity() {
        return () -> {
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            if (authentication == null) {
                // no user associated (not in a request context, e.g. sqs queue handling)
                throw new NullAuthenticationException();
            }
            Object principal = authentication.getPrincipal();
            if (principal instanceof User testAuthUser) {
                TestUserProperties testUserProperties = testUserMap.get(testAuthUser.getUsername());
                return new IdentityAttributes(
                        testUserProperties.getInternalId(),
                        testUserProperties.getExternalId(),
                        testUserProperties.getEmail(),
                        testUserProperties.getTin());
            }
            throw new RuntimeException("Unexpected authentication principal of type " + principal.getClass());
        };
    }
}
