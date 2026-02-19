package gov.irs.directfile.api.config;

import java.net.URI;

import com.amazon.sqs.javamessaging.ProviderConfiguration;
import com.amazon.sqs.javamessaging.SQSConnectionFactory;
import jakarta.jms.Connection;
import jakarta.jms.JMSException;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sqs.SqsClient;

@Configuration
@EnableConfigurationProperties({S3ConfigurationProperties.class, MessageQueueConfigurationProperties.class})
@Slf4j
@AllArgsConstructor
public class SqsMessageQueueConfiguration {

    private final AwsCredentialsProvider awsCredentialsProvider;

    @Bean
    SqsClient getSqs(
            S3ConfigurationProperties s3ConfigurationProperties,
            MessageQueueConfigurationProperties messageQueueConfigurationProperties) {
        return SqsClient.builder()
                .region(Region.of(s3ConfigurationProperties.getRegion()))
                .credentialsProvider(awsCredentialsProvider)
                .endpointOverride(URI.create(messageQueueConfigurationProperties.getEndpoint()))
                .build();
    }

    @Bean
    public SQSConnectionFactory sqsConnectionFactory(SqsClient sqsClient) {
        return new SQSConnectionFactory(new ProviderConfiguration(), sqsClient);
    }

    @Bean
    public Connection jmsConnection(SQSConnectionFactory sqsConnectionFactory) throws JMSException {
        Connection connection = sqsConnectionFactory.createConnection();
        connection.start(); // Start the connection to enable message delivery
        log.info("CONNECTED TO SQS");
        return connection;
    }
}
