package gov.irs.directfile.submit.config;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;

import gov.irs.directfile.submit.mocks.FakeSynchronousDocumentStorageService;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@TestConfiguration
public class SynchronousS3TestConfiguration {

    /**
     * Tell Spring to use this fake implementation for the s3 storage service.
     *
     *
     * */
    @Bean
    @Primary
    ISynchronousDocumentStoreService fakeSynchronousS3StorageService() {
        return new FakeSynchronousDocumentStorageService();
    }
}
