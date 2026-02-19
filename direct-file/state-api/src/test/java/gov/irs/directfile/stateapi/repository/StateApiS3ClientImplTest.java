package gov.irs.directfile.stateapi.repository;

import java.util.UUID;
import java.util.concurrent.CompletableFuture;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.core.async.AsyncResponseTransformer;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.encryption.s3.S3AsyncEncryptionClient;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.configuration.S3ConfigurationProperties;
import gov.irs.directfile.stateapi.configuration.XmlSanitizedConfigurationProperties;
import gov.irs.directfile.stateapi.exception.StateApiException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(SpringExtension.class)
@ContextConfiguration(
        classes = {XmlSanitizedConfigurationProperties.class, S3ConfigurationProperties.class},
        initializers = {ConfigDataApplicationContextInitializer.class})
@EnableConfigurationProperties(value = {XmlSanitizedConfigurationProperties.class, S3ConfigurationProperties.class})
class StateApiS3ClientImplTest {

    StateApiS3ClientImpl stateApiS3Client;

    @Mock
    private S3AsyncClient s3Client;

    @Mock
    private S3AsyncEncryptionClient s3EncryptionClient;

    @Autowired
    private S3ConfigurationProperties s3ConfigurationProperties;

    @Autowired
    private XmlSanitizedConfigurationProperties xmlSanitizedConfigurationProperties;

    @BeforeEach
    void setup() {
        stateApiS3Client = new StateApiS3ClientImpl(
                s3Client, s3EncryptionClient, s3ConfigurationProperties, xmlSanitizedConfigurationProperties);
    }

    @Test
    void givenXmlExistsInS3_whenGetTaxReturnXml_thenReturnsSanitizedXml() {
        // GIVEN
        int taxFilingYear = 2024;
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "submissionId";

        GetObjectResponse getObjectResponse = GetObjectResponse.builder()
                .contentLength(1024L) // Add any mock data for your test
                .build();

        // Create ResponseBytes object (use byte array or any content to simulate)
        byte[] data = taxReturnXml.getBytes();
        ResponseBytes<GetObjectResponse> responseBytes = ResponseBytes.fromByteArray(getObjectResponse, data);

        // Create CompletableFuture and complete it with the simulated response
        CompletableFuture<ResponseBytes<GetObjectResponse>> s3ClientResponse =
                CompletableFuture.completedFuture(responseBytes);

        when(s3EncryptionClient.getObject(any(GetObjectRequest.class), any(AsyncResponseTransformer.class)))
                .thenReturn(s3ClientResponse);

        // WHEN
        Mono<String> responseMono = stateApiS3Client.getTaxReturnXml(taxFilingYear, taxReturnId, submissionId);

        // THEN
        StepVerifier.create(responseMono)
                .expectNextMatches(xmlString -> xmlString.equals(sanitizedTaxReturnXml))
                .verifyComplete();

        // verify the bucket and object key used to fetch XML from S3
        ArgumentCaptor<GetObjectRequest> objectRequestArgumentCaptor = ArgumentCaptor.captor();
        verify(s3EncryptionClient)
                .getObject(objectRequestArgumentCaptor.capture(), any(AsyncResponseTransformer.class));
        GetObjectRequest getObjectRequest = objectRequestArgumentCaptor.getValue();
        assertEquals(s3ConfigurationProperties.getTaxReturnXmlBucketName(), getObjectRequest.bucket());

        String expectedObjectKey =
                taxFilingYear + "/taxreturns/" + taxReturnId + "/submissions/" + submissionId + ".xml";
        assertEquals(expectedObjectKey, getObjectRequest.key());
    }

    @Test
    void givenXmlExistsInS3AndS3PrefixIsSet_whenGetTaxReturnXml_thenGetsXmlAtThePrefixedLocation() {
        // GIVEN
        int taxFilingYear = 2024;
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "submissionId";
        String s3Prefix = "testPrefix";
        s3ConfigurationProperties.setPrefix(s3Prefix);

        GetObjectResponse getObjectResponse = GetObjectResponse.builder()
                .contentLength(1024L) // Add any mock data for your test
                .build();

        // Create ResponseBytes object (use byte array or any content to simulate)
        byte[] data = taxReturnXml.getBytes();
        ResponseBytes<GetObjectResponse> responseBytes = ResponseBytes.fromByteArray(getObjectResponse, data);

        // Create CompletableFuture and complete it with the simulated response
        CompletableFuture<ResponseBytes<GetObjectResponse>> s3ClientResponse =
                CompletableFuture.completedFuture(responseBytes);

        when(s3EncryptionClient.getObject(any(GetObjectRequest.class), any(AsyncResponseTransformer.class)))
                .thenReturn(s3ClientResponse);

        // WHEN
        Mono<String> responseMono = stateApiS3Client.getTaxReturnXml(taxFilingYear, taxReturnId, submissionId);

        // THEN
        StepVerifier.create(responseMono)
                .expectNextMatches(xmlString -> xmlString.equals("<xml>"))
                .verifyComplete();

        // verify the bucket and object key used to fetch XML from S3
        ArgumentCaptor<GetObjectRequest> objectRequestArgumentCaptor = ArgumentCaptor.captor();
        verify(s3EncryptionClient)
                .getObject(objectRequestArgumentCaptor.capture(), any(AsyncResponseTransformer.class));
        GetObjectRequest getObjectRequest = objectRequestArgumentCaptor.getValue();
        assertEquals(s3ConfigurationProperties.getTaxReturnXmlBucketName(), getObjectRequest.bucket());

        String expectedObjectKey =
                s3Prefix + "/" + taxFilingYear + "/taxreturns/" + taxReturnId + "/submissions/" + submissionId + ".xml";
        assertEquals(expectedObjectKey, getObjectRequest.key());
    }

    @Test
    @Disabled("OSS changes alter the output")
    void givenErrorWhenFetchingXmlFromS3_whenGetTaxReturnXml_thenReturnsError() {
        // GIVEN
        int taxFilingYear = 2024;
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "submissionId";

        when(s3EncryptionClient.getObject(any(GetObjectRequest.class), any(AsyncResponseTransformer.class)))
                .thenReturn(Mono.error(new RuntimeException()).toFuture());

        // WHEN
        Mono<String> responseMono = stateApiS3Client.getTaxReturnXml(taxFilingYear, taxReturnId, submissionId);

        // THEN
        StepVerifier.create(responseMono)
                .expectErrorMatches(e -> e instanceof StateApiException
                        && e.getMessage().equals(StateApiErrorCode.E_INTERNAL_SERVER_ERROR.toString()))
                .verify();
    }

    @Test
    @Disabled("Skip - OSS changes alter the output")
    void givenErrorWhenSanitizingXml_whenGetTaxReturnXml_thenReturnsError() {
        // GIVEN
        int taxFilingYear = 2024;
        UUID taxReturnId = UUID.randomUUID();
        String submissionId = "submissionId";

        GetObjectResponse getObjectResponse = GetObjectResponse.builder()
                .contentLength(1024L) // Add any mock data for your test
                .build();

        // Create ResponseBytes object (use byte array or any content to simulate)
        byte[] data = "".getBytes();
        ResponseBytes<GetObjectResponse> responseBytes = ResponseBytes.fromByteArray(getObjectResponse, data);

        // Create CompletableFuture and complete it with the simulated response
        CompletableFuture<ResponseBytes<GetObjectResponse>> s3ClientResponse =
                CompletableFuture.completedFuture(responseBytes);

        when(s3EncryptionClient.getObject(any(GetObjectRequest.class), any(AsyncResponseTransformer.class)))
                .thenReturn(s3ClientResponse);

        // WHEN
        Mono<String> responseMono = stateApiS3Client.getTaxReturnXml(taxFilingYear, taxReturnId, submissionId);

        // THEN
        StepVerifier.create(responseMono)
                .expectErrorMatches(e -> e instanceof StateApiException
                        && e.getMessage().equals(StateApiErrorCode.E_INTERNAL_SERVER_ERROR.toString()))
                .verify();
    }

    final String taxReturnXml = "<xml>";
    final String sanitizedTaxReturnXml = "<xml>";
}
