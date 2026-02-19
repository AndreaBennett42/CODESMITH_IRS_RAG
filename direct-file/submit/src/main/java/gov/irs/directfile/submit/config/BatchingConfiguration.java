package gov.irs.directfile.submit.config;

import java.time.Clock;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import gov.irs.directfile.submit.BatchingProperties;

@Configuration
@Slf4j
public class BatchingConfiguration {
    @Value("${submit.batching.batchSize}")
    private int batchSize;

    @Value("${submit.batching.batchTimeoutMilliseconds}")
    private long batchTimeoutMilliseconds;

    @Bean
    public BatchingProperties batchingProperties() {

        BatchingProperties properties = BatchingProperties.builder()
                .batchTimeoutMilliseconds(batchTimeoutMilliseconds)
                .maxBatchSize(batchSize)
                .build();
        log.info("Starting app with Batching Properties " + properties);
        return properties;
    }

    @Bean
    public Clock batchingClock() {
        return Clock.systemUTC();
    }
}
