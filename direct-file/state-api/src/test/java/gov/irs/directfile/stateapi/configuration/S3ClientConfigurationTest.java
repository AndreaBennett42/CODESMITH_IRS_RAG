package gov.irs.directfile.stateapi.configuration;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

import static org.assertj.core.api.Assertions.assertThat;

class S3ClientConfigurationTest {

    private final ApplicationContextRunner applicationContextRunner = new ApplicationContextRunner()
            .withPropertyValues(
                    "aws.s3.accessKey=test",
                    "aws.s3.secretKey=test",
                    "aws.s3.region=us-west-2",
                    "aws.s3.assumeRoleArn=test",
                    "aws.s3.assumeRoleDurationSeconds=3600",
                    "aws.s3.assumeRoleSessionName=test",
                    "aws.s3.endPoint=http://directfile.test",
                    "aws.s3.certBucketName=test",
                    "aws.s3.taxReturnXmlBucketName=test",
                    "aws.s3.s3-kms-wrapping-key-arn=test",
                    "aws.s3.default-credentials-provider-chain-enabled=false",
                    "direct-file.local-encryption.local-wrapping-key=test",
                    "aws.kmsEndpoint=http://directfile.test",
                    "aws.region=us-west-2")
            .withUserConfiguration(
                    S3ClientConfiguration.class,
                    EncryptionClientConfiguration.class,
                    AwsCredentialsConfiguration.class);

    @Test
    void s3AsyncClientBeanIsCreated() {
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(S3AsyncClient.class)).isNotNull();
        });
    }

    @Test
    void s3AsyncEncryptionClientBeanIsCreatedWhenUsingAwsProfile() {
        applicationContextRunner
                .withPropertyValues("spring.profiles.active=aws")
                .run((context) -> {
                    assertThat(context.getBean(S3AsyncEncryptionClient.class)).isNotNull();
                });
    }

    @Test
    void s3AsyncEncryptionClientBeanIsCreatedWhenNotUsingAwsProfile() {
        applicationContextRunner.run((context) -> {
            assertThat(context.getBean(S3AsyncEncryptionClient.class)).isNotNull();
        });
    }
}
