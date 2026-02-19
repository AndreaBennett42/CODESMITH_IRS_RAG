package gov.irs.directfile.emailservice.services;

import com.amazon.sqs.javamessaging.ProviderConfiguration;
import com.amazon.sqs.javamessaging.SQSConnection;
import com.amazon.sqs.javamessaging.SQSConnectionFactory;
import com.amazon.sqs.javamessaging.SQSSession;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.annotation.PreDestroy;
import jakarta.jms.*;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;

import gov.irs.directfile.emailservice.config.EmailServiceConfigurationProperties;
import gov.irs.directfile.emailservice.listeners.SendEmailMessageQueueListener;

@Slf4j
@Service
@ConditionalOnProperty(value = "email-service.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP2"},
        justification = "Initial SpotBugs Setup")
@SuppressWarnings({"PMD.CloseResource", "PMD.UnusedPrivateMethod"})
public class SqsConnectionSetupService {
    private final EmailServiceConfigurationProperties configProps;
    private final SqsClient sqsClient;
    private final EmailRecordKeepingService emailRecordKeepingService;
    ISendService sender;
    private SQSConnection connection;

    @SneakyThrows
    public SqsConnectionSetupService(
            EmailServiceConfigurationProperties configProps,
            SqsClient sqsClient,
            EmailRecordKeepingService emailRecordKeepingService,
            ISendService sender)
            throws JMSException {
        this.configProps = configProps;
        this.sqsClient = sqsClient;
        this.emailRecordKeepingService = emailRecordKeepingService;
        this.sender = sender;

        this.setup();
    }

    public void setup() throws JMSException {
        SQSConnectionFactory connectionFactory = new SQSConnectionFactory(new ProviderConfiguration(), sqsClient);
        log.info("Creating SQS queue connection for email service");
        this.connection = connectionFactory.createConnection();
        log.info("Creating SQS queue session for email service");
        Session session = connection.createSession(SQSSession.UNORDERED_ACKNOWLEDGE);

        Queue queue = session.createQueue(configProps.getMessageQueue().getSendEmailQueue());
        MessageConsumer consumer = session.createConsumer(queue);

        log.info("Setting SendEmailMessageQueueListener as queue consumer");
        consumer.setMessageListener(
                new SendEmailMessageQueueListener(configProps, sender, sqsClient, emailRecordKeepingService));

        connection.start();

        log.info("CONNECTED TO SQS");
    }

    @PreDestroy
    private void cleanup() throws JMSException {
        if (this.connection != null) {
            connection.stop();
            log.info("SQS connection stopped");
            connection.close();
            log.info("SQS connection closed");
        }
    }
}
