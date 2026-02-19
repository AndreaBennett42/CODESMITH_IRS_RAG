package gov.irs.directfile.stateapi.repository;

import java.io.IOException;
import java.util.UUID;

import com.github.dockerjava.zerodep.shaded.org.apache.hc.core5.http.HttpStatus;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import gov.irs.directfile.stateapi.configuration.DirectFileEndpointProperties;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.model.TaxReturnStatus;

@ExtendWith(SpringExtension.class)
@ContextConfiguration(initializers = ConfigDataApplicationContextInitializer.class)
@EnableConfigurationProperties(value = {DirectFileEndpointProperties.class})
class DirectFileClientImplTest {

    static MockWebServer webServer;

    private DirectFileClient repo;

    @Autowired
    DirectFileEndpointProperties dfep;

    @BeforeAll
    static void setUp() throws IOException {
        webServer = new MockWebServer();
        webServer.start();
    }

    @AfterAll
    static void tearDown() throws IOException {
        webServer.shutdown();
    }

    @BeforeEach
    void init() {
        String url = String.format("http://localhost:%s", webServer.getPort());
        dfep.setBackendUrl(url);
        repo = new DirectFileClientImpl(dfep);
    }

    @Test
    void testGetStatus_Success() throws Exception {
        String resp = "{\"status\":\"Accepted\", \"exists\":true}";
        UUID id = UUID.randomUUID();
        int taxYear = 2022;

        webServer.enqueue(new MockResponse().setBody(resp).addHeader("Content-Type", "application/json"));
        String submissionId = "submissionId";
        Mono<TaxReturnStatus> resultMono = repo.getStatus(taxYear, id, submissionId);

        StepVerifier.create(resultMono)
                .expectNextMatches(s -> s.status().equalsIgnoreCase("accepted"))
                .verifyComplete();
    }

    @Test
    void testGetStatus_WithException() throws Exception {
        UUID id = UUID.randomUUID();
        int taxYear = 2022;

        webServer.enqueue(new MockResponse().setResponseCode(HttpStatus.SC_INTERNAL_SERVER_ERROR));
        String submissionId = "submissionId";
        Mono<TaxReturnStatus> accepted = repo.getStatus(taxYear, id, submissionId);

        StepVerifier.create(accepted).expectError(StateApiException.class).verify();
    }
}
