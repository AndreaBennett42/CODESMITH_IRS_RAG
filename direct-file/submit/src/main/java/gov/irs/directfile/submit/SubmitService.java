package gov.irs.directfile.submit;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import jakarta.annotation.PostConstruct;
import jakarta.transaction.Transactional;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.service.*;

@Service
@EnableScheduling
@Transactional
@Slf4j
@SuppressWarnings({"PMD.ExcessiveParameterList", "PMD.SignatureDeclareThrowsException", "PMD.AvoidCatchingThrowable"})
public class SubmitService {
    private final Config config;
    private final SqsConnectionSetupService sqsConnectionSetupService;

    private final UserSubmissionConsumer userSubmissionConsumer;

    @SneakyThrows
    public SubmitService(
            Config configuration,
            SqsConnectionSetupService sqsConnectionSetupService,
            UserSubmissionConsumer userSubmissionConsumer) {
        this.config = configuration;
        this.sqsConnectionSetupService = sqsConnectionSetupService;
        this.userSubmissionConsumer = userSubmissionConsumer;
        log.info("Setting up the MeF connection");
    }

    @PostConstruct
    public void setup() throws Exception {
        createDirectories();

        if (config.isRunnerDisabledForTesting()) {
            log.info("exiting setup before creating services because this is a test tun");
            return;
        }

        if (config.getMessageQueue().isSqsMessageHandlingEnabled()) {
            sqsConnectionSetupService.setup(userSubmissionConsumer);
        }
    }

    private void createDirectories() {
        try {
            log.info("Attempting to create directories for data writes");
            new DirectoryCreator(config).CreateDirectories();
        } catch (IOException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }
}
