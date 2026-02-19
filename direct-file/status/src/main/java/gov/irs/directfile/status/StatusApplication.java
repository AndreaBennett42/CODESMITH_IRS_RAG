package gov.irs.directfile.status;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Scope;

import gov.irs.mef.services.msi.LoginClient;
import gov.irs.mef.services.msi.LogoutClient;
import gov.irs.mef.services.transmitter.mtom.GetAcksMTOMClient;

@SpringBootApplication
@ConfigurationPropertiesScan
public class StatusApplication {
    @Bean
    @Scope("prototype")
    public LoginClient loginClient() {
        return new LoginClient();
    }

    @Bean
    @Scope("prototype")
    public GetAcksMTOMClient ackClient() {
        return new GetAcksMTOMClient();
    }

    @Bean
    @Scope("prototype")
    public LogoutClient logoutClient() {
        return new LogoutClient();
    }

    public static void main(String[] args) {
        SpringApplication.run(StatusApplication.class, args);
    }
}
