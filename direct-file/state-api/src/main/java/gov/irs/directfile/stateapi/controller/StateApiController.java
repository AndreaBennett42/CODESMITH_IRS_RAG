package gov.irs.directfile.stateapi.controller;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import dev.openfeature.sdk.Client;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.dto.StateProfileDTO;
import gov.irs.directfile.stateapi.encryption.JwtVerifier;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.exception.StateNotExistException;
import gov.irs.directfile.stateapi.model.*;
import gov.irs.directfile.stateapi.service.StateApiService;

@RestController
@RequestMapping("/state-api")
@Validated
@Slf4j
@SuppressWarnings({"PMD.MissingOverride", "PMD.ExcessiveParameterList"})
@RequiredArgsConstructor
public class StateApiController implements StateApi {
    public static final String SESSION_KEY = "SESSION-KEY";
    public static final String INITIALIZATION_VECTOR = "INITIALIZATION-VECTOR";
    public static final String AUTHENTICATION_TAG = "AUTHENTICATION-TAG";

    private final StateApiService svc;
    private final Client featureFlagClient;

    private static final String X_HEADER = "x-header";

    @PostMapping("/authorization-code")
    @ResponseStatus(HttpStatus.CREATED)
    public Mono<ResponseEntity<AuthCodeResponse>> createAuthorizationCode(
            @Valid @RequestBody AuthCodeRequest ac, HttpServletRequest request) {

        log.info("enter createAuthorizationCode()...");

        String taxReturnId = ac.getTaxReturnUuid().toString();
        String submissionId = ac.getSubmissionId();

        return svc.createAuthorizationCode(ac)
                .map(uuid -> ResponseEntity.ok(
                        AuthCodeResponse.builder().authCode(uuid).build()))
                .doOnSuccess(response -> {
                    log.info(
                            "createAuthorizationCode() succeeded for taxYear: {}, stateCode: {}, taxReturnId: {}",
                            ac.getTaxYear(),
                            ac.getStateCode(),
                            taxReturnId);
                })
                .onErrorResume(e -> {
                    log.error(
                            "createAuthorizationCode() failed for taxYear: {}, stateCode: {}, taxReturnId: {}, submissionId: {}, {}, error: {}",
                            ac.getTaxYear(),
                            ac.getStateCode(),
                            taxReturnId,
                            submissionId,
                            e.getClass().getName(),
                            e.getMessage());
                    if (e instanceof StateApiException stateApiException) {
                        // TODO: Http Status should be dependent on the error code. For now, since
                        // StateApiExceptionHandler previously returned 500 for any unhandled exception, this keeps the
                        // status the same
                        return Mono.just(ResponseEntity.internalServerError()
                                .body(AuthCodeResponse.builder()
                                        .errorCode(stateApiException.getErrorCode())
                                        .build()));
                    } else {
                        return Mono.error(e);
                    }
                });
    }

    /**
     **
     * Prerequisite:**
     * The state will generate a key pair and send the public key (e.g., AZ.cer,
     * NY.cer) to the IRS.
     *
     * DF State-Tax-Api (Server):**
     * 1. Generate a secret (32 bytes).
     * 2. Generate an initialization vector (IV) value (12 bytes).
     * 3. Encrypt the tax return XML with the secret from step 1 and the IV from
     * step 2 using the AES-256 GCM algorithm (plus Authentication Tag).
     * 4. Base64 encode the encrypted tax return XML from step 3.
     * 5. Encrypt the 32-byte secret using the public key provided by the state
     * (RSA).
     * 6. Base64 encode the encrypted secret from step 5.
     * 7. Base64 encode the IV from step 2.
     * 8. Base64 encode the Authentication Tag from step 3.
     * 9. Send HTTP response to the State-App with:
     * a) Header: SESSION-KEY=value from step 6
     * b) Header: INITIALIZATION_VECTOR=value from step 7
     * c) Header: AUTHENTICATION-TAG=value from step 8
     * d) Body: value from step 4.
     *
     * State-App Side (Client):**
     * 1. Extract SESSION-KEY from the HTTP response header.
     * 2. Base64 decode the SESSION-KEY.
     * 3. Decrypt the decoded SESSION-KEY with its private key (RSA) (in byte[]).
     * 4. Extract INITIALIZATION_VECTOR (IV) from the HTTP response header.
     * 5. Base64 decode the IV.
     * 6. Extract AUTHENTICATION_TAG from the HTTP response header.
     * 7. Base64 decode AUTHENTICATION_TAG.
     * 6. Decrypt the XML data from the response body using the secret from step 3,
     * the IV from step 5 using the AES-256 GCM algorithm, verify with
     * Authentication Tag from step 7.
     *
     * @param authorizationHeader
     *            the JWT Bearer token
     * @return the encrypted and BASE 64 encoded tax return xml along with two
     *         headers in HTTP Response: SESSION-KEY, INITIALIZATION-VECTOR and
     *         AUTHENTICATION-TAG.
     *         http status code: 200 for all responses.
     *
     *         if success, response as:
     *         {"status": "success", "taxReturn": "encoded-encrypted-data"}
     *         (taxReturn includes return status and xml data, status can be
     *         "accepted", "rejected", "pending". Sample value:
     *         {"status":"accepted", "xml":"return-data"})
     *         if failed, response as:
     *         {"status": "error", "error": "E_AUTHORIZATION_CODE_EXPIRED"}
     */
    @GetMapping("/export-return")
    public Mono<ResponseEntity<ExportResponse>> exportReturn(
            @RequestHeader("Authorization") String authorizationHeader, HttpServletRequest request) {

        log.info("Enter exportReturn()...");

        AtomicInteger taxYear = new AtomicInteger();
        AtomicReference<String> stateCode = new AtomicReference<>();
        stateCode.set(getStateCode(request));
        AtomicReference<String> taxReturnId = new AtomicReference<>();
        AtomicReference<String> submissionId = new AtomicReference<>();
        taxReturnId.set("");
        submissionId.set("");

        if (Boolean.FALSE.equals(this.featureFlagClient.getBooleanValue("export-return", Boolean.TRUE))) {
            log.info("Export return is disabled via configuration property. No action taken");
            return Mono.just(ResponseEntity.ok(handleErrors(
                    null,
                    taxYear.get(),
                    stateCode.get(),
                    taxReturnId.get(),
                    submissionId.get(),
                    new StateApiException(StateApiErrorCode.E_STATE_API_DISABLED))));
        }

        String prefix = "Bearer ";

        if (!authorizationHeader.startsWith(prefix)) {
            return Mono.just(ResponseEntity.ok(handleErrors(
                    null,
                    taxYear.get(),
                    stateCode.get(),
                    taxReturnId.get(),
                    submissionId.get(),
                    new StateApiException(StateApiErrorCode.E_BEARER_TOKEN_MISSING))));
        }

        String jwtToken = authorizationHeader.substring(prefix.length());

        final String accountId;

        try {
            accountId = JwtVerifier.getAccountId(jwtToken);
        } catch (StateApiException e) {
            return Mono.just(ResponseEntity.ok(
                    handleErrors(null, taxYear.get(), stateCode.get(), taxReturnId.get(), submissionId.get(), e)));
        }

        return svc.verifyJwtSignature(jwtToken, accountId)
                .flatMap(saCode -> {
                    stateCode.set(saCode.getStateCode());
                    return svc.authorize(saCode).flatMap(entity -> {
                        taxReturnId.set(entity.getTaxReturnUuid().toString());
                        taxYear.set(entity.getTaxYear());
                        return svc.retrieveTaxReturnXml(
                                        taxYear.get(), entity.getTaxReturnUuid(), entity.getSubmissionId())
                                .flatMap(trXml -> {
                                    submissionId.set(trXml.submissionId());
                                    return svc.retrieveExportedFacts(
                                                    entity.getSubmissionId(), stateCode.get(), accountId)
                                            .flatMap(exportedFacts -> {
                                                TaxReturnToExport trExpt = new TaxReturnToExport(
                                                        trXml.status(),
                                                        trXml.submissionId(),
                                                        trXml.xml(),
                                                        exportedFacts);
                                                return svc.encryptTaxReturn(trExpt, accountId)
                                                        .map(ed -> ResponseEntity.ok()
                                                                .header(SESSION_KEY, ed.encodedSecret())
                                                                .header(INITIALIZATION_VECTOR, ed.encodedIV())
                                                                .header(
                                                                        AUTHENTICATION_TAG,
                                                                        ed.encodedAuthenticationTag())
                                                                .body(ExportResponse.builder()
                                                                        .status("success")
                                                                        .taxReturn(ed.encodedAndEncryptedData())
                                                                        .build()));
                                            });
                                });
                    });
                })
                .doOnSuccess(export -> {
                    log.info(
                            "exportReturn succeeded for taxYear: {}, accountId: {}, stateCode: {}, taxReturnId: {}, submissionId: {}",
                            taxYear.get(),
                            accountId,
                            stateCode.get(),
                            taxReturnId.get(),
                            submissionId.get());
                })
                .onErrorResume(e -> Mono.just(ResponseEntity.ok(handleErrors(
                        accountId, taxYear.get(), stateCode.get(), taxReturnId.get(), submissionId.get(), e))));
    }

    @GetMapping("/state-profile")
    public Mono<ResponseEntity<StateProfileDTO>> getStateProfile(
            @RequestParam String stateCode, HttpServletRequest request) {

        log.info("enter getStateProfile()...");

        return svc.lookupStateProfile(stateCode)
                .map(ResponseEntity::ok)
                .doOnSuccess(export -> {
                    log.info("getStateProfile completes successfully for stateCode {}", stateCode);
                })
                .onErrorResume(StateNotExistException.class, e -> {
                    log.info("No state profile found for {}, {}", stateCode, e.getMessage());
                    return Mono.just(ResponseEntity.noContent().build());
                })
                .onErrorResume(Mono::error);
    }

    private ExportResponse handleErrors(
            String accountId, int taxYear, String stateCode, String taxReturnId, String submissionId, Throwable t) {
        StateApiErrorCode errorCode;

        log.error(
                "exportReturn failed for taxYear: {}, accountId: {}, stateCode: {}, taxReturnId: {}, submissionId: {}, {}, error: {}",
                taxYear,
                accountId,
                stateCode,
                taxReturnId,
                submissionId,
                t.getClass().getName(),
                t.getMessage());

        if (t instanceof StateApiException e) {
            // Hide specific error codes from the client, return internal server error
            // instead
            if (e.getErrorCode() == StateApiErrorCode.E_CERTIFICATE_NOT_FOUND
                    || e.getErrorCode() == StateApiErrorCode.E_TAX_RETURN_NOT_FOUND) {
                errorCode = StateApiErrorCode.E_INTERNAL_SERVER_ERROR;
            } else if (e.getErrorCode() == StateApiErrorCode.E_EXPORTED_FACTS_DISABLED) {
                errorCode = StateApiErrorCode.E_INTERNAL_SERVER_ERROR;
                log.info("{}: {}", e.getErrorCode(), e.getMessage());
            } else {
                errorCode = e.getErrorCode();
            }
        } else {
            errorCode = StateApiErrorCode.E_INTERNAL_SERVER_ERROR;
        }

        return ExportResponse.builder().status("error").error(errorCode.name()).build();
    }

    private String getStateCode(HttpServletRequest request) {

        String stateInfo = request.getHeader(X_HEADER);
        String stateCd = "";
        // if (stateInfo != null && !stateInfo.isEmpty()) {
        if (StringUtils.isNotBlank(stateInfo)) {
            stateCd = stateInfo.substring(0, 2);
        } else {
            log.info(X_HEADER + " does not exist or is empty");
        }
        return stateCd;
    }
}
