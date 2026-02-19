package gov.irs.directfile.submit;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import gov.irs.directfile.submit.config.*;

@SpringBootApplication
@EnableConfigurationProperties({
    Config.class,
})
public class SubmitApplication {

    public static void main(String[] args) {
        System.setProperty("spring.devtools.restart.enabled", "false");
        SpringApplication.run(SubmitApplication.class, args);
    }
}
