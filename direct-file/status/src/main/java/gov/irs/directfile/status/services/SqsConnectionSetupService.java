package gov.irs.directfile.status.services;

import com.amazon.sqs.javamessaging.ProviderConfiguration;
import com.amazon.sqs.javamessaging.SQSConnection;
import com.amazon.sqs.javamessaging.SQSConnectionFactory;
import com.amazon.sqs.javamessaging.SQSSession;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.annotation.PreDestroy;
import jakarta.jms.JMSException;
import jakarta.jms.MessageConsumer;
import jakarta.jms.Queue;
import jakarta.jms.Session;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;

import gov.irs.directfile.status.config.MessageQueueConfiguration;

@Service
@ConditionalOnProperty(value = "status.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@Slf4j
@SuppressFBWarnings(value = "CT_CONSTRUCTOR_THROW", justification = "Java 21 update")
@SuppressWarnings({"PMD.CloseResource", "PMD.UnusedPrivateMethod"})
@EnableConfigurationProperties(MessageQueueConfiguration.class)
public class SqsConnectionSetupService {
    private final SqsClient sqsClient;
    private final String pendingSubmissionQueue;
    private final MessageQueueListenerService messageQueueListenerService;
    private SQSConnection connection;

    SqsConnectionSetupService(
            SqsClient sqsClient,
            MessageQueueConfiguration messageQueueConfiguration,
            MessageQueueListenerService messageQueueListenerService)
            throws JMSException {
        this.sqsClient = sqsClient;
        this.pendingSubmissionQueue = messageQueueConfiguration.getPendingSubmissionQueue();
        this.messageQueueListenerService = messageQueueListenerService;

        initializeSQSConnection();
    }

    private void initializeSQSConnection() throws JMSException {
        SQSConnectionFactory connectionFactory = new SQSConnectionFactory(new ProviderConfiguration(), sqsClient);
        connection = connectionFactory.createConnection();

        Session session = connection.createSession(false, SQSSession.UNORDERED_ACKNOWLEDGE);
        Queue queue = session.createQueue(pendingSubmissionQueue);
        MessageConsumer consumer = session.createConsumer(queue);
        consumer.setMessageListener(messageQueueListenerService);

        connection.start();
        log.info("CONNECTED TO SQS");
    }

    @PreDestroy
    private void cleanup() throws JMSException {
        if (connection != null) {
            connection.stop();
            log.info("SQS connection stopped");
            connection.close();
            log.info("SQS connection closed");
        }
    }
}
