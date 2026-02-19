package gov.irs.directfile.api.taxreturn.submissions;

import com.amazon.sqs.javamessaging.SQSSession;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import jakarta.annotation.PreDestroy;
import jakarta.jms.Connection;
import jakarta.jms.JMSException;
import jakarta.jms.MessageConsumer;
import jakarta.jms.MessageListener;
import jakarta.jms.Queue;
import jakarta.jms.Session;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;

@Slf4j
@Service
@SuppressFBWarnings(value = "CT_CONSTRUCTOR_THROW", justification = "Java 21 update")
@SuppressWarnings(value = {"PMD.ExcessiveParameterList", "PMD.CloseResource"})
@ConditionalOnProperty(value = "direct-file.aws.messageQueue.sqs-message-handling-enabled", havingValue = "true")
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class SqsConnectionSetupService {
    private final Connection connection;
    private final String submissionConfirmationQueue;
    private final String sendEmailQueue;
    private final ConfirmationQueueListenerService confirmationQueueListenerService;
    private final String submissionStatusQueue;
    private final StatusQueueListenerService statusQueueListenerService;
    private final S3NotificationEventQueueListenerService s3NotificationEventQueueListenerService;
    private final DlqS3NotificationEventQueueListenerService dlqS3NotificationEventQueueListenerService;
    private final String dlqS3NotificationEventQueueListenerServiceQueue;
    private final String s3NotificationEventQueueListenerServiceQueue;

    SqsConnectionSetupService(
            Connection connection,
            MessageQueueConfigurationProperties messageQueueConfigurationProperties,
            ConfirmationQueueListenerService confirmationQueueListenerService,
            StatusQueueListenerService statusQueueListenerService,
            S3NotificationEventQueueListenerService s3NotificationEventQueueListenerService,
            DlqS3NotificationEventQueueListenerService dlqS3NotificationEventQueueListenerService)
            throws JMSException {

        this.connection = connection;
        this.sendEmailQueue = messageQueueConfigurationProperties.getSendEmailQueue();
        this.submissionConfirmationQueue = messageQueueConfigurationProperties.getSubmissionConfirmationQueue();
        this.confirmationQueueListenerService = confirmationQueueListenerService;
        this.submissionStatusQueue = messageQueueConfigurationProperties.getStatusChangeQueue();
        this.statusQueueListenerService = statusQueueListenerService;
        this.s3NotificationEventQueueListenerService = s3NotificationEventQueueListenerService;
        this.dlqS3NotificationEventQueueListenerService = dlqS3NotificationEventQueueListenerService;
        this.s3NotificationEventQueueListenerServiceQueue =
                messageQueueConfigurationProperties.getS3NotificationEventQueue();
        this.dlqS3NotificationEventQueueListenerServiceQueue =
                messageQueueConfigurationProperties.getDlqS3NotificationEventQueue();

        this.initializeSQSConnection();
    }

    private void initializeSQSConnection() throws JMSException {
        Session session = connection.createSession(false, SQSSession.UNORDERED_ACKNOWLEDGE);

        // set up submission confirmation queue listener
        Queue submissionConfirmationQueue = session.createQueue(this.submissionConfirmationQueue);
        MessageConsumer confirmationConsumer = session.createConsumer(submissionConfirmationQueue);
        confirmationConsumer.setMessageListener(this.confirmationQueueListenerService);

        // set up submission status queue listener
        Queue submissionStatusQueue = session.createQueue(this.submissionStatusQueue);
        MessageConsumer statusConsumer = session.createConsumer(submissionStatusQueue);
        statusConsumer.setMessageListener(this.statusQueueListenerService);

        // set up s3 notification event queue listener
        Queue s3NotificationEventQueue = session.createQueue(this.s3NotificationEventQueueListenerServiceQueue);
        MessageConsumer s3NotificationEventConsumer = session.createConsumer(s3NotificationEventQueue);
        s3NotificationEventConsumer.setMessageListener(this.s3NotificationEventQueueListenerService);

        // set up send email queue onto which we will enqueue messages
        session.createQueue(this.sendEmailQueue);

        // start setting up DLQ
        setUpDlq(
                this.s3NotificationEventQueueListenerServiceQueue,
                dlqS3NotificationEventQueueListenerServiceQueue,
                dlqS3NotificationEventQueueListenerService);
        // end setting up DLQ

        connection.start();
        log.info("CONNECTED TO SQS");
    }

    private void setUpDlq(String sourceQueueName, String dlqName, MessageListener dlqListener) throws JMSException {
        log.info("Setup Dlq, queueName {}, dlqName{}", sourceQueueName, dlqName);
        Session sessionDlq = connection.createSession(false, SQSSession.AUTO_ACKNOWLEDGE);
        Queue queueDlq = sessionDlq.createQueue(dlqName);
        MessageConsumer dlqConsumer = sessionDlq.createConsumer(queueDlq);
        dlqConsumer.setMessageListener(dlqListener);
    }

    @SuppressWarnings("PMD.UnusedPrivateMethod")
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
