package gov.irs.directfile.submit.service.startup.runners;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

import gov.irs.directfile.submit.service.UserSubmissionBatchProcessor;

@Component
@Slf4j
@Order(2)
/**
 * When the application starts, this class calls processOldBatches() to pick up any unprocessed
 * submissions in Document Storage
 *
 * The ApplicationRunner interface allows us to run one-time logic after an application has
 * started.
 * Reference Docs:
 * https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/#features.spring-application.command-line-runner
 *
 * In this case, we want to process any old batches in S3 after the application has started.
 * */
public class BatchProcessingApplicationStartRunner implements ApplicationRunner {
    private final UserSubmissionBatchProcessor userSubmissionBatchProcessor;

    public BatchProcessingApplicationStartRunner(UserSubmissionBatchProcessor userSubmissionBatchProcessor) {
        this.userSubmissionBatchProcessor = userSubmissionBatchProcessor;
    }

    @Override
    public void run(ApplicationArguments args) throws Exception {
        log.info("=== Processing Old Batches now that application has started. ===");
        userSubmissionBatchProcessor.processOldBatches();
    }
}
