package gov.irs.directfile.stateapi.controller;

import java.security.Security;
import java.util.*;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import dev.openfeature.sdk.Client;
import lombok.SneakyThrows;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.mock.web.MockHttpServletRequest;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.audit.AuditService;
import gov.irs.directfile.stateapi.dto.StateProfileDTO;
import gov.irs.directfile.stateapi.encryption.JwtSigner;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.exception.StateApiExportedFactsDisabledException;
import gov.irs.directfile.stateapi.exception.StateNotExistException;
import gov.irs.directfile.stateapi.model.*;
import gov.irs.directfile.stateapi.service.StateApiService;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@SuppressWarnings("null")
public class StateApiControllerTest {

    @InjectMocks
    private StateApiController controller;

    @Mock
    private Client featureFlagClient;

    @Mock
    private StateApiService svc;

    @InjectMocks
    private static MockHttpServletRequest mockRequest;

    private static StateProfile sp;

    private final Logger logger = (Logger) LoggerFactory.getLogger(AuditService.class);
    private ListAppender<ILoggingEvent> listAppender;

    private static final String X_FORWARDED_FOR = "X-Forwarded-For";
    private static final String TEST_IP_ADDR1 = "10.1.2.1";
    private static final String TEST_IP_ADDR2 = "10.1.2.2";
    private static final String REMOTE_IP_ADDR = "10.1.2.3";

    @BeforeAll
    static void init() {
        Security.addProvider(new BouncyCastleProvider());
        mockRequest = new MockHttpServletRequest();
        Security.addProvider(new BouncyCastleProvider());
        mockRequest.addHeader("x-header", "AZStateAUP");
        mockRequest.setRemoteAddr(REMOTE_IP_ADDR);

        sp = new StateProfile();
        sp.setAccountId("111111");
        sp.setStateCode("FS");
        sp.setAcceptedOnly(true);
        sp.setTaxSystemName("sth");
        sp.setLandingUrl("url");
        sp.setDepartmentOfRevenueUrl("url");
        sp.setFilingRequirementsUrl("url");
    }

    @Test
    public void testCreateAuthorizationCode_Success() throws Exception {

        UUID authCode = UUID.randomUUID();
        var expectedResponseBody = AuthCodeResponse.builder().authCode(authCode).build();

        AuthCodeRequest ac = new AuthCodeRequest(UUID.randomUUID(), "tin", 2022, "FS", "submissionId");

        when(svc.createAuthorizationCode(any())).thenReturn(Mono.just(authCode));

        Mono<ResponseEntity<AuthCodeResponse>> responseMono = controller.createAuthorizationCode(ac, mockRequest);
        StepVerifier.create(responseMono)
                .expectNextMatches(response -> Objects.equals(response.getBody(), expectedResponseBody))
                .verifyComplete();
    }

    @Test
    public void testCreateAuthorizationCodeWithNullTin_Success() throws Exception {

        UUID authCode = UUID.randomUUID();
        var expectedResponseBody = AuthCodeResponse.builder().authCode(authCode).build();

        AuthCodeRequest ac = new AuthCodeRequest(UUID.randomUUID(), null, 2022, "FS", "submissionId");

        when(svc.createAuthorizationCode(any())).thenReturn(Mono.just(authCode));

        Mono<ResponseEntity<AuthCodeResponse>> responseMono = controller.createAuthorizationCode(ac, mockRequest);
        StepVerifier.create(responseMono)
                .expectNextMatches(response -> Objects.equals(response.getBody(), expectedResponseBody))
                .verifyComplete();
    }

    @Test
    public void testCreateAuthorizationCode_UnhandledException() throws Exception {
        AuthCodeRequest ac = new AuthCodeRequest(UUID.randomUUID(), "tin", 2022, "FS", "submissionId");
        when(svc.createAuthorizationCode(any()))
                .thenReturn(Mono.error(new RuntimeException("Something bad happened!")));

        Mono<ResponseEntity<AuthCodeResponse>> responseMono = controller.createAuthorizationCode(ac, mockRequest);
        StepVerifier.create(responseMono)
                .expectErrorMatches(e -> e instanceof RuntimeException)
                .verify();
    }

    @Test
    public void testCreateAuthorizationCode_Exception() throws Exception {
        StateApiErrorCode errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING;
        var expectedResponse = AuthCodeResponse.builder().errorCode(errorCode).build();

        AuthCodeRequest ac = new AuthCodeRequest(UUID.randomUUID(), "tin", 2022, "FS", "submissionId");
        when(svc.createAuthorizationCode(any())).thenReturn(Mono.error(new StateApiException(errorCode)));

        Mono<ResponseEntity<AuthCodeResponse>> responseMono = controller.createAuthorizationCode(ac, mockRequest);
        StepVerifier.create(responseMono)
                .expectNextMatches(response -> response.getStatusCode().equals(HttpStatus.INTERNAL_SERVER_ERROR)
                        && Objects.equals(response.getBody(), expectedResponse))
                .verifyComplete();
    }

    @Test
    public void testExportReturn_Success() throws Exception {
        ClientJwtClaim claim = new ClientJwtClaim();
        String accountId = "1234567890";
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        ac.setSubmissionId("submissionId");
        claim.setAccountId(accountId);
        claim.setAuthorizationCode(authorizationCode.toString());
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "NY");

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;

        String xml = """
                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                """;

        var exportedFacts = new HashMap<String, Object>();
        exportedFacts.put("filers", List.of(Map.of("firstName", "Tom", "lastName", "Smith")));

        TaxReturnXml trData = new TaxReturnXml("accepted", "the-submission-id", xml);
        TaxReturnToExport trExport =
                new TaxReturnToExport("accepted", "the-submission-id", trData.xml(), exportedFacts);
        EncryptData encryptData = new EncryptData("the-secret", "the-iv", "the-data", "the-authentication-tag");

        mockExportReturnEnabled(true);
        when(svc.verifyJwtSignature(jwtToken, accountId)).thenReturn(Mono.just(saCode));
        when(svc.authorize(saCode)).thenReturn(Mono.just(ac));
        when(svc.retrieveTaxReturnXml(ac.getTaxYear(), ac.getTaxReturnUuid(), ac.getSubmissionId()))
                .thenReturn(Mono.just(trData));
        when(svc.retrieveExportedFacts(ac.getSubmissionId(), saCode.getStateCode(), accountId))
                .thenReturn(Mono.just(exportedFacts));
        when(svc.encryptTaxReturn(trExport, accountId)).thenReturn(Mono.just(encryptData));

        Mono<ResponseEntity<ExportResponse>> exportMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(exportMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && export.getHeaders().get("SESSION-KEY").get(0).equals("the-secret")
                        && export.getHeaders()
                                .get("INITIALIZATION-VECTOR")
                                .get(0)
                                .equals("the-iv")
                        && export.getHeaders().get("AUTHENTICATION-TAG").get(0).equals("the-authentication-tag")
                        && export.getBody().getStatus().equals("success")
                        && export.getBody().getTaxReturn().equals("the-data"))
                .verifyComplete();
    }

    @Test
    public void testExportReturnWithSpouse_Success() throws Exception {
        ClientJwtClaim claim = new ClientJwtClaim();
        String accountId = "1234567890";
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        ac.setSubmissionId("submissionId");
        claim.setAccountId(accountId);
        claim.setAuthorizationCode(authorizationCode.toString());
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "NY");

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;

        String xml = """
                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                """;

        var exportedFacts = new HashMap<String, Object>();
        exportedFacts.put(
                "filers",
                List.of(
                        Map.of("firstName", "Tom", "lastName", "Smith"),
                        Map.of("firstName", "Jane", "lastName", "Smith")));

        TaxReturnXml trData = new TaxReturnXml("accepted", "the-submission-id", xml);
        TaxReturnToExport trExport =
                new TaxReturnToExport("accepted", "the-submission-id", trData.xml(), exportedFacts);
        EncryptData encryptData = new EncryptData("the-secret", "the-iv", "the-data", "the-authentication-tag");

        mockExportReturnEnabled(true);
        when(svc.verifyJwtSignature(jwtToken, accountId)).thenReturn(Mono.just(saCode));
        when(svc.authorize(saCode)).thenReturn(Mono.just(ac));
        when(svc.retrieveTaxReturnXml(ac.getTaxYear(), ac.getTaxReturnUuid(), ac.getSubmissionId()))
                .thenReturn(Mono.just(trData));
        when(svc.retrieveExportedFacts(ac.getSubmissionId(), saCode.getStateCode(), accountId))
                .thenReturn(Mono.just(exportedFacts));
        when(svc.encryptTaxReturn(trExport, accountId)).thenReturn(Mono.just(encryptData));

        Mono<ResponseEntity<ExportResponse>> exportMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(exportMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && Objects.requireNonNull(export.getHeaders().get("SESSION-KEY"))
                                .getFirst()
                                .equals("the-secret")
                        && Objects.requireNonNull(export.getHeaders().get("INITIALIZATION-VECTOR"))
                                .getFirst()
                                .equals("the-iv")
                        && Objects.requireNonNull(export.getHeaders().get("AUTHENTICATION-TAG"))
                                .getFirst()
                                .equals("the-authentication-tag")
                        && Objects.requireNonNull(export.getBody()).getStatus().equals("success")
                        && export.getBody().getTaxReturn().equals("the-data"))
                .verifyComplete();
    }

    @Test
    public void testExportReturn_Unauthorized() throws Exception {
        String accountId = "123456";
        UUID authorizationCode = UUID.randomUUID();

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "FS");

        mockExportReturnEnabled(true);
        when(svc.verifyJwtSignature(jwtToken, accountId)).thenReturn(Mono.just(saCode));
        when(svc.authorize(saCode)).thenThrow(new StateApiException(StateApiErrorCode.E_AUTHORIZATION_CODE_NOT_EXIST));

        Mono<ResponseEntity<ExportResponse>> exportMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(exportMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && export.getBody().getStatus().equals("error"))
                .verifyComplete();
    }

    @Test
    public void testExportReturn_BearerTokenMissing() throws Exception {
        String accountId = "123456";
        UUID authorizationCode = UUID.randomUUID();

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;
        authHeader = authHeader.replaceAll("Bearer", "");

        mockExportReturnEnabled(true);
        Mono<ResponseEntity<ExportResponse>> exportMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(exportMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && export.getBody().getStatus().equals("error"))
                .verifyComplete();
    }

    @Test
    public void testExportReturn_WithRetrieveException() throws Exception {
        String accountId = "123456";

        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "FS");

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;

        mockExportReturnEnabled(true);
        when(svc.verifyJwtSignature(jwtToken, accountId)).thenReturn(Mono.just(saCode));
        when(svc.authorize(saCode)).thenReturn(Mono.just(ac));
        when(svc.retrieveTaxReturnXml(ac.getTaxYear(), ac.getTaxReturnUuid(), ac.getSubmissionId()))
                .thenThrow(new StateApiException(StateApiErrorCode.E_TAX_RETURN_NOT_FOUND));

        Mono<ResponseEntity<ExportResponse>> encryptDataMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(encryptDataMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && Objects.requireNonNull(export.getBody()).getStatus().equals("error"))
                .verifyComplete();
    }

    @Test
    public void testExportReturn_WithEncryptException() throws Exception {
        String accountId = "the-account-id";
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        String xml =
                """
                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                        """;
        var exportedFacts = new HashMap<String, Object>();
        exportedFacts.put("filers", List.of(Map.of("firstName", "Tom", "lastName", "Smith")));

        TaxReturnXml trData = new TaxReturnXml("accepted", "the-submission-id", xml);
        TaxReturnToExport trExport =
                new TaxReturnToExport("accepted", "the-submission-id", trData.xml(), exportedFacts);

        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "FS");
        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;

        mockExportReturnEnabled(true);
        when(svc.verifyJwtSignature(jwtToken, accountId)).thenReturn(Mono.just(saCode));
        when(svc.authorize(saCode)).thenReturn(Mono.just(ac));
        when(svc.retrieveTaxReturnXml(ac.getTaxYear(), ac.getTaxReturnUuid(), ac.getSubmissionId()))
                .thenReturn(Mono.just(trData));
        when(svc.retrieveExportedFacts(ac.getSubmissionId(), saCode.getStateCode(), accountId))
                .thenReturn(Mono.just(exportedFacts));
        when(svc.encryptTaxReturn(trExport, accountId))
                .thenThrow(new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR));

        Mono<ResponseEntity<ExportResponse>> encryptDataMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(encryptDataMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && export.getBody().getStatus().equals("error"))
                .verifyComplete();
    }

    @SneakyThrows
    @Test
    public void givenExportReturnRequest_whenStateApiExportedFactsDisabledException_thenReturnsInternalServerError() {
        String accountId = "123456";

        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        StateAndAuthCode saCode = new StateAndAuthCode(authorizationCode.toString(), "FS");
        String xml = """
                <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                """;
        TaxReturnXml trData = new TaxReturnXml("accepted", "the-submission-id", xml);

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;

        mockExportReturnEnabled(true);
        when(svc.verifyJwtSignature(jwtToken, accountId)).thenReturn(Mono.just(saCode));
        when(svc.authorize(saCode)).thenReturn(Mono.just(ac));
        when(svc.retrieveTaxReturnXml(ac.getTaxYear(), ac.getTaxReturnUuid(), ac.getSubmissionId()))
                .thenReturn(Mono.just(trData));
        when(svc.retrieveExportedFacts(ac.getSubmissionId(), saCode.getStateCode(), accountId))
                .thenThrow(new StateApiExportedFactsDisabledException());

        Mono<ResponseEntity<ExportResponse>> exportResponse = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(exportResponse)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && Objects.requireNonNull(export.getBody()).getStatus().equals("error")
                        && export.getBody().getError().equals(StateApiErrorCode.E_INTERNAL_SERVER_ERROR.toString()))
                .verifyComplete();
    }

    @Test
    public void testGetStateProfile_Success() {

        String stateCode = "FS";

        List<StateRedirect> stateRedirects = new ArrayList<>();
        var stateRedirect1 = mock(StateRedirect.class);
        when(stateRedirect1.getRedirectUrl()).thenReturn("http://redirect.state.system/1");
        var stateRedirect2 = mock(StateRedirect.class);
        when(stateRedirect2.getRedirectUrl()).thenReturn("http://redirect.state.system/2");

        stateRedirects.add(stateRedirect1);
        stateRedirects.add(stateRedirect2);

        List<StateLanguage> stateLanguages = new ArrayList<>();
        var stateLanguage1 = mock(StateLanguage.class);
        when(stateLanguage1.getDfLanguageCode()).thenReturn("en");
        when(stateLanguage1.getStateLanguageCode()).thenReturn("english");

        var stateLanguage2 = mock(StateLanguage.class);
        when(stateLanguage2.getDfLanguageCode()).thenReturn("es");
        when(stateLanguage2.getStateLanguageCode()).thenReturn("spanish");

        stateLanguages.add(stateLanguage1);
        stateLanguages.add(stateLanguage2);

        var spDto = new StateProfileDTO(sp, stateRedirects, stateLanguages);
        when(svc.lookupStateProfile(any())).thenReturn(Mono.just(spDto));

        Mono<ResponseEntity<StateProfileDTO>> aMono = controller.getStateProfile(stateCode, mockRequest);
        StepVerifier.create(aMono)
                .expectNextMatches(returnedSpDto -> returnedSpDto
                                .getStatusCode()
                                .equals(HttpStatus.OK)
                        && returnedSpDto.getBody().stateCode().equals(stateCode)
                        && returnedSpDto
                                .getBody()
                                .redirectUrls()
                                .containsAll(List.of(stateRedirect1.getRedirectUrl(), stateRedirect2.getRedirectUrl()))
                        && "english".equals(returnedSpDto.getBody().languages().get("en"))
                        && "spanish".equals(returnedSpDto.getBody().languages().get("es"))
                        && "url".equals(returnedSpDto.getBody().departmentOfRevenueUrl())
                        && "url".equals(returnedSpDto.getBody().filingRequirementsUrl()))
                .verifyComplete();
    }

    @Test
    public void testGetStateProfile_NotExist() {

        String stateCode = "FS";
        when(svc.lookupStateProfile(any()))
                .thenReturn(Mono.error(new StateNotExistException(StateApiErrorCode.E_STATE_NOT_EXIST)));

        Mono<ResponseEntity<StateProfileDTO>> aMono = controller.getStateProfile(stateCode, mockRequest);

        StepVerifier.create(aMono)
                .expectNextMatches(
                        response -> response.getStatusCode().equals(HttpStatus.NO_CONTENT) && !response.hasBody())
                .verifyComplete();
    }

    @Test
    public void testGetStateProfile_Exception() {

        String stateCode = "FS";
        when(svc.lookupStateProfile(any()))
                .thenReturn(Mono.error(new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR)));

        Mono<ResponseEntity<StateProfileDTO>> aMono = controller.getStateProfile(stateCode, mockRequest);

        StepVerifier.create(aMono)
                .expectErrorMatches(
                        e -> e instanceof StateApiException && e.getMessage().equals("E_INTERNAL_SERVER_ERROR"))
                .verify();
    }

    @Test
    void exportReturnReturnsEmptyResponseWhenNotEnabled() throws Exception {
        mockExportReturnEnabled(false);

        ClientJwtClaim claim = new ClientJwtClaim();
        String accountId = "1234567890";
        UUID authorizationCode = UUID.randomUUID();
        AuthorizationCode ac = new AuthorizationCode();
        ac.setAuthorizationCode(authorizationCode);
        ac.setTaxReturnUuid(UUID.randomUUID());
        ac.setTaxYear(2022);
        claim.setAccountId(accountId);
        claim.setAuthorizationCode(authorizationCode.toString());

        String jwtToken = createJwtToken(accountId, authorizationCode);
        String authHeader = "Bearer " + jwtToken;

        mockExportReturnEnabled(false);
        Mono<ResponseEntity<ExportResponse>> responseMono = controller.exportReturn(authHeader, mockRequest);

        StepVerifier.create(responseMono)
                .expectNextMatches(export -> export.getStatusCode() == HttpStatus.OK
                        && export.getBody().getStatus().equals("error")
                        && export.getBody().getError().contains("E_STATE_API_DISABLED"))
                .verifyComplete();

        verifyNoInteractions(svc);
    }

    private String createJwtToken(String accountId, UUID authorizationCode) throws Exception {
        String header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
        String payload =
                "{\"iss\":\"" + accountId + "\",\"sub\":\"" + authorizationCode.toString() + "\",\"iat\":1632736766}";

        // Create the JWT
        String jwtToken = JwtSigner.createJwt(header, payload, "src/test/resources/certificates/fakestate.key");

        return "Bearer " + jwtToken;
    }

    private void mockExportReturnEnabled(boolean exportReturnEnabled) {
        when(featureFlagClient.getBooleanValue(any(), any())).thenReturn(exportReturnEnabled);
    }
}
