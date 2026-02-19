package gov.irs.directfile.submit.service.interfaces;

import gov.irs.directfile.submit.config.Config;

public interface IService {
    void Setup(Config config) throws Throwable;
}
