package gov.irs.directfile.stateapi;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.servers.Server;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableScheduling;

import gov.irs.directfile.stateapi.configuration.CertificationOverrideProperties;
import gov.irs.directfile.stateapi.configuration.DirectFileEndpointProperties;
import gov.irs.directfile.stateapi.configuration.FeatureFlagsConfigurationProperties;
import gov.irs.directfile.stateapi.configuration.S3ConfigurationProperties;
import gov.irs.directfile.stateapi.configuration.XmlSanitizedConfigurationProperties;

@SpringBootApplication
@EnableCaching
@EnableScheduling
@EnableConfigurationProperties({
    FeatureFlagsConfigurationProperties.class,
    XmlSanitizedConfigurationProperties.class,
    S3ConfigurationProperties.class,
    DirectFileEndpointProperties.class,
    CertificationOverrideProperties.class
})
@OpenAPIDefinition(
        info = @Info(title = "State Tax API", description = "The State Tax API", version = "1.0.1"),
        servers = {
            @Server(url = "https://df.services.irs.gov", description = "Prod Server"),
            @Server(url = "https://df.alt.services.irs.gov", description = "ATS Server"),
            @Server(url = "http://locahost:8081/state-api", description = "Local Development")
        },
        security = {@SecurityRequirement(name = "PreauthenticatedHeader")})
public class StateApiApp {

    public static void main(String[] args) {
        SpringApplication.run(StateApiApp.class, args);
    }
}
