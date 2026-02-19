package gov.irs.directfile.submit.actions;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import gov.irs.a2a.mef.mefheader.TestCdType;
import gov.irs.mef.services.ServiceContext;
import gov.irs.mef.services.data.ETIN;

import gov.irs.directfile.submit.config.Config;

@Slf4j
@SuppressWarnings("PMD.LiteralsFirstInComparisons")
@Getter
public class ActionContext {
    private Config config;
    private ServiceContext serviceContext;

    public ActionContext(Config config) {
        this.config = config;
        TestCdType testCd = TestCdType.T;
        if (config.isProd()) {
            log.info("This is a production release pointing at real MeF! Pointing toolkit to prod");
            testCd = TestCdType.P;
        } else {
            log.info("Non-prod release, pointing at PRE PROD!");
        }

        CreateServiceContext(config.getEtin(), config.getAsid(), testCd);
    }

    private void CreateServiceContext(String etinString, String asidString, TestCdType testCd) {
        ETIN etin = new ETIN(etinString);
        serviceContext = new ServiceContext(etin, asidString, testCd);
    }
}
