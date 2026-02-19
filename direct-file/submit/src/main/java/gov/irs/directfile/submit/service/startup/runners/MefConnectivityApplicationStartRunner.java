package gov.irs.directfile.submit.service.startup.runners;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

import gov.irs.directfile.submit.service.polling.MeFHealthPoller;

@Component
@Slf4j
@Order(1)
public class MefConnectivityApplicationStartRunner implements ApplicationRunner {
    private final MeFHealthPoller meFHealthPoller;

    public MefConnectivityApplicationStartRunner(MeFHealthPoller meFHealthPoller) {
        this.meFHealthPoller = meFHealthPoller;
    }

    @Override
    public void run(ApplicationArguments args) throws Exception {
        boolean isMefOnline = meFHealthPoller.performMefConnectivityCheck();

        if (isMefOnline) {
            log.info("MeF Client is online. Initial MeF login and logout were successful");
        }
    }
}
