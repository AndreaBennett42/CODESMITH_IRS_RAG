package gov.irs.directfile.stateapi.repository;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.UUID;

import com.fasterxml.jackson.databind.ObjectMapper;
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

import gov.irs.directfile.models.StateOrProvince;
import gov.irs.directfile.stateapi.configuration.DirectFileEndpointProperties;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.exception.StateApiExportedFactsDisabledException;
import gov.irs.directfile.stateapi.model.GetStateExportedFactsResponse;
import gov.irs.directfile.stateapi.model.TaxReturnStatus;

@ExtendWith(SpringExtension.class)
@ContextConfiguration(initializers = ConfigDataApplicationContextInitializer.class)
@EnableConfigurationProperties(value = {DirectFileEndpointProperties.class})
public class DirectFileBackendServiceTest {

    static MockWebServer webServer;

    @Autowired
    private DirectFileEndpointProperties dfep;

    private DirectFileBackendService dfbService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private UUID id = UUID.randomUUID();
    private int taxYear = 2023;
    private String submissionId = "submissionId";
    private String accountId = "123456";

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
        dfbService = new DirectFileBackendService(dfep);
    }

    @Test
    void testGetExportedFacts_Success() throws Exception {

        var exportedFacts = new HashMap<String, Object>();
        var filers = createFilerList();
        exportedFacts.put("filers", filers);
        var resp = new GetStateExportedFactsResponse(exportedFacts);

        String json = objectMapper.writeValueAsString(resp);

        webServer.enqueue(new MockResponse().setBody(json).addHeader("Content-Type", "application/json"));

        Mono<GetStateExportedFactsResponse> resultMono =
                dfbService.getExportedFacts(submissionId, StateOrProvince.NY.name(), accountId);

        StepVerifier.create(resultMono)
                .expectNextMatches(s -> s.exportedFacts().get("filers").equals(filers))
                .verifyComplete();
    }

    @Test
    void testGetExportedFacts_WithException() throws Exception {

        webServer.enqueue(new MockResponse().setResponseCode(HttpStatus.SC_INTERNAL_SERVER_ERROR));
        Mono<GetStateExportedFactsResponse> result =
                dfbService.getExportedFacts(submissionId, StateOrProvince.NY.name(), accountId);

        StepVerifier.create(result).expectError(StateApiException.class).verify();
    }

    @Test
    void testGetExportedFacts_withMethodNotAllowedResponse_throwsCustomException() {
        webServer.enqueue(new MockResponse().setResponseCode(HttpStatus.SC_METHOD_NOT_ALLOWED));
        Mono<GetStateExportedFactsResponse> result =
                dfbService.getExportedFacts(submissionId, StateOrProvince.NY.name(), accountId);
        StepVerifier.create(result)
                .expectError(StateApiExportedFactsDisabledException.class)
                .verify();
    }

    @Test
    void testGetStatus_Success() throws Exception {

        String resp = "{\"status\":\"Pending\", \"exists\":true}";

        webServer.enqueue(new MockResponse().setBody(resp).addHeader("Content-Type", "application/json"));

        Mono<TaxReturnStatus> resultMono = dfbService.getStatus(taxYear, id, submissionId);

        StepVerifier.create(resultMono)
                .expectNextMatches(s -> s.status().equalsIgnoreCase("pending"))
                .verifyComplete();
    }

    @Test
    void testGetStatus_WithException() throws Exception {

        webServer.enqueue(new MockResponse().setResponseCode(HttpStatus.SC_INTERNAL_SERVER_ERROR));
        Mono<TaxReturnStatus> accepted = dfbService.getStatus(taxYear, id, submissionId);

        StepVerifier.create(accepted).expectError(StateApiException.class).verify();
    }

    private ArrayList<HashMap<String, Object>> createFilerList() {
        var filers = new ArrayList<HashMap<String, Object>>();
        var filer1 = new HashMap<String, Object>();
        filer1.put("firstName", "Samuel");
        filer1.put("lastName", "Smith");
        filer1.put("dateOfBirth", "1985-09-29");
        filer1.put("isPrimaryFiler", Boolean.TRUE);
        filer1.put("tin", "100-01-1234");
        filers.add(filer1);
        return filers;
    }
}
