package gov.irs.directfile.submit.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import gov.irs.mef.ApplicationContext;

import gov.irs.directfile.submit.actions.ActionContext;

@Configuration
@Slf4j
public class ActionContextConfig {
    @Bean
    public ActionContext actionContext(Config config) {
        log.info("Setting the MeF toolkit");
        ApplicationContext.setToolkitHome(config.getToolkit());
        log.info("Toolkit set");
        return new ActionContext(config);
    }
}
