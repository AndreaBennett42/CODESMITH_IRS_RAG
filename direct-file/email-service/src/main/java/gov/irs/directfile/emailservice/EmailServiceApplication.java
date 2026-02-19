package gov.irs.directfile.emailservice;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import gov.irs.directfile.emailservice.config.EmailServiceConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties({EmailServiceConfigurationProperties.class})
public class EmailServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(EmailServiceApplication.class, args);
    }
}
