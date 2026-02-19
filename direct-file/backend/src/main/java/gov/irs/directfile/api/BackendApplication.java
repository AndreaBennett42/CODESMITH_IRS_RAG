package gov.irs.directfile.api;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.servers.Server;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

import gov.irs.directfile.api.config.*;

@SpringBootApplication
@EnableScheduling
@EnableConfigurationProperties({
    StateApiEndpointProperties.class,
    StateApiFeatureFlagProperties.class,
    StatusEndpointProperties.class,
    SubmitEndpointProperties.class,
})
@OpenAPIDefinition(
        info = @Info(title = "Direct File API", description = "The Direct File API", version = "1.0.1"),
        servers = {
            @Server(url = "http://localhost:8080${server.servlet.context-path}", description = "Local development"),
        })
public class BackendApplication {
    public static void main(String[] args) {
        SpringApplication.run(BackendApplication.class, args);
    }
}
