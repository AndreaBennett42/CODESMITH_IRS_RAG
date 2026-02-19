package gov.irs.boot.autoconfigure.boilerplate;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

// @SpringBootTest
class BoilerplateStarterTests {

    private final ApplicationContextRunner contextRunner =
            new ApplicationContextRunner().withConfiguration(AutoConfigurations.of(BoilerplateAutoConfiguration.class));

    @Test
    void contextLoads() {
        // @SpringBootTest will attempt to load the ApplicationContext which will fail if the
        // SampleStarterConfigurationProperties fails to load

    }
}
