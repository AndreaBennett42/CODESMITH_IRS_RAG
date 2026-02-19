package gov.irs.directfile.submit.config;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.domain.SubmissionBatch;

@Configuration
public class ActionQueueConfig {

    @Bean
    public ActionQueue actions() {
        return new ActionQueue();
    }

    @Bean
    public Set<SubmissionBatch> inProgressBatches() {
        return ConcurrentHashMap.newKeySet();
    }
}
