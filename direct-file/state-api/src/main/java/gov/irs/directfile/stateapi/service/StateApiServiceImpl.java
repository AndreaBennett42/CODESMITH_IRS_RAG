package gov.irs.directfile.stateapi.service;

import java.io.FileNotFoundException;
import java.security.PublicKey;
import java.security.Security;
import java.security.cert.CertificateExpiredException;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbusds.jose.shaded.gson.Gson;
import com.nimbusds.jose.shaded.gson.GsonBuilder;
import io.micrometer.common.util.StringUtils;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.util.encoders.Base64;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.authorization.AuthorizationTokenClaims;
import gov.irs.directfile.stateapi.authorization.AuthorizationTokenService;
import gov.irs.directfile.stateapi.configuration.CertificationOverrideProperties;
import gov.irs.directfile.stateapi.dto.StateProfileDTO;
import gov.irs.directfile.stateapi.encryption.JwtVerifier;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.model.*;
import gov.irs.directfile.stateapi.repository.*;
import gov.irs.directfile.stateapi.repository.facts.ExportedFactsClient;

import static gov.irs.directfile.stateapi.encryption.Encryptor.*;
import static gov.irs.directfile.stateapi.repository.DirectFileClient.STATUS_ERROR;
import static gov.irs.directfile.stateapi.repository.DirectFileClient.STATUS_REJECTED;

@Service
@Slf4j
@SuppressWarnings({
    "PMD.UnusedPrivateMethod",
    "PMD.LiteralsFirstInComparisons",
    "PMD.UselessParentheses",
    "PMD.PreserveStackTrace"
})
public class StateApiServiceImpl implements StateApiService {
    @Value("${authorization-code.expires-interval-seconds: 600}")
    private int authorizationCodeExpiresInterval;

    @Value("${direct-file.exported-facts.enabled: false}")
    private boolean exported_facts_enabled;

    @Autowired
    private AuthorizationCodeRepository acRepo;

    @Autowired
    private AuthorizationTokenService authorizationTokenService;

    @Autowired
    private DirectFileClient dfClient;

    @Autowired
    private ExportedFactsClient efClient;

    @Autowired
    private CachedDataService cachedDS;

    @Autowired
    private ObjectMapper mapper;

    @Autowired
    private StateApiS3Client stateApiS3Client;

    @Autowired
    CertificationOverrideProperties certProperties;

    @PostConstruct
    private void init() {
        Security.addProvider(new BouncyCastleProvider());
    }

    @Override
    public Mono<UUID> createAuthorizationCode(AuthCodeRequest acq) {
        log.info("enter createAuthorizationCode()...");

        // The `dfClient.getStatus()` function returns the status along with an indicator of whether the XML exists in
        // the S3 bucket. However, it's important to consider the issue of eventual consistency, which refers to the
        // delay in propagating S3 objects. While we may verify the existence of the object in the west region, there's
        // a possibility that when performing the `export-tax-return` operation, retrieval occurs from the east region,
        // potentially resulting in a 'tax-return-not-found' error.
        return getAcceptedOnlyFlag(acq.getStateCode())
                .zipWith(
                        dfClient.getStatus(acq.getTaxYear(), acq.getTaxReturnUuid(), acq.getSubmissionId()),
                        (acceptedOnly, status) -> {
                            final boolean isAccepted =
                                    status.status().equalsIgnoreCase(DirectFileClient.STATUS_ACCEPTED);
                            final boolean isPending = status.status().equalsIgnoreCase(DirectFileClient.STATUS_PENDING);

                            final boolean submissionOk =
                                    status.exists() && (isAccepted || (!acceptedOnly && isPending));
                            if (submissionOk) {
                                final AuthorizationCode ac = new AuthorizationCode();
                                final UUID code = UUID.randomUUID();
                                ac.setAuthorizationCode(code);
                                ac.setTaxReturnUuid(acq.getTaxReturnUuid());
                                ac.setTaxYear(acq.getTaxYear());
                                Timestamp expireAt =
                                        Timestamp.from(Instant.now().plusSeconds(authorizationCodeExpiresInterval));
                                ac.setExpiresAt(expireAt);
                                ac.setStateCode(acq.getStateCode());
                                ac.setSubmissionId(acq.getSubmissionId());

                                return acRepo.save(ac).thenReturn(code).onErrorMap(e -> {
                                    log.error(
                                            "createAuthorizationCode() failed for taxReturnId={}, submissionId={}, taxYear={}. Exception: {}. Error: {}",
                                            acq.getTaxReturnUuid(),
                                            acq.getSubmissionId(),
                                            acq.getTaxYear(),
                                            e.getClass().getName(),
                                            e.getMessage());
                                    throw new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                                });
                            } else {
                                final StateApiErrorCode errorCode;
                                if (!status.exists()) {
                                    errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_FOUND;
                                } else if (acceptedOnly) {
                                    errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED;
                                } else {
                                    errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING;
                                }

                                log.error(
                                        "createAuthorizationCode() failed for taxReturnId={}, submissionId={}, taxYear={}. Error: {}",
                                        acq.getTaxReturnUuid(),
                                        acq.getSubmissionId(),
                                        acq.getTaxYear(),
                                        errorCode.name());
                                throw new StateApiException(errorCode);
                            }
                        })
                .flatMap(uuidMono -> uuidMono.map(uuid -> uuid));
    }

    @Override
    public Mono<String> generateAuthorizationToken(AuthCodeRequest acRequest) {
        log.info("Generating authorization token");

        // The `dfClient.getStatus()` function returns the status along with an indicator of whether the XML exists in
        // the S3 bucket. However, it's important to consider the issue of eventual consistency, which refers to the
        // delay in propagating S3 objects. While we may verify the existence of the object in the west region, there's
        // a possibility that when performing the `export-tax-return` operation, retrieval occurs from the east region,
        // potentially resulting in a 'tax-return-not-found' error.
        return getAcceptedOnlyFlag(acRequest.getStateCode())
                .zipWith(
                        dfClient.getStatus(
                                acRequest.getTaxYear(), acRequest.getTaxReturnUuid(), acRequest.getSubmissionId()),
                        (acceptedOnly, status) -> {
                            var isAccepted = status.status().equalsIgnoreCase(DirectFileClient.STATUS_ACCEPTED);
                            var isPending = status.status().equalsIgnoreCase(DirectFileClient.STATUS_PENDING);

                            boolean submissionOk = status.exists() && (isAccepted || (!acceptedOnly && isPending));
                            if (submissionOk) {
                                return authorizationTokenService
                                        .generateAndEncrypt(
                                                mapper.convertValue(acRequest, AuthorizationTokenClaims.class))
                                        .onErrorMap(e -> {
                                            log.error(
                                                    "generateAuthorizationToken() failed for taxReturnId={}, submissionId={}, taxYear={}. Exception: {}. Error: {}",
                                                    acRequest.getTaxReturnUuid(),
                                                    acRequest.getSubmissionId(),
                                                    acRequest.getTaxYear(),
                                                    e.getClass().getName(),
                                                    e.getMessage());
                                            throw new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                                        });
                            } else {
                                StateApiErrorCode errorCode;
                                if (!status.exists()) {
                                    errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_FOUND;
                                } else if (acceptedOnly) {
                                    errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED;
                                } else {
                                    errorCode = StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING;
                                }

                                log.error(
                                        "generateAuthorizationToken() failed for taxReturnId={}, submissionId={}, taxYear={}. Error: {}",
                                        acRequest.getTaxReturnUuid(),
                                        acRequest.getSubmissionId(),
                                        acRequest.getTaxYear(),
                                        errorCode.name());
                                throw new StateApiException(errorCode);
                            }
                        })
                .flatMap(jwtMono -> jwtMono.map(token -> token));
    }

    @Override
    public Mono<StateAndAuthCode> verifyJwtSignature(String jwtToken, String accountId) {
        log.info("enter verifyJwtSignature()...");

        return getStateProfile(accountId)
                .zipWhen(sp -> {
                    return retrievePublicKeyFromCert(sp);
                })
                .map(tuple -> {
                    try {
                        ClientJwtClaim claim = JwtVerifier.verifyJwt(jwtToken, tuple.getT2());
                        return new StateAndAuthCode(
                                claim.getAuthorizationCode(), tuple.getT1().getStateCode());
                    } catch (FileNotFoundException e) {
                        log.error(
                                "verifyJwtSignature() failed, cannot locate the public key file, {}, error: {}",
                                e.getClass(),
                                e.getMessage());
                        throw new StateApiException(StateApiErrorCode.E_CERTIFICATE_NOT_FOUND);
                    } catch (CertificateExpiredException e) {
                        log.error(
                                "verifyJwtSignature() failed, the certificate ({}) for {} has expired, {}, error: {}",
                                tuple.getT1().getCertLocation(),
                                tuple.getT1().getAccountId(),
                                e.getClass(),
                                e.getMessage());
                        throw new StateApiException(StateApiErrorCode.E_CERTIFICATE_EXPIRED);
                    } catch (Exception e) {
                        log.error(
                                "verifyJwtSignature() failed, {}, error: {}",
                                e.getClass().getName(),
                                e.getMessage());
                        throw new StateApiException(StateApiErrorCode.E_JWT_VERIFICATION_FAILED);
                    }
                });
    }

    @Override
    public Mono<AuthorizationCode> authorize(StateAndAuthCode saCode) {
        log.info("enter authorize()...");

        UUID authorizationCode;

        try {
            authorizationCode = UUID.fromString(saCode.getAuthorizationCode());
        } catch (IllegalArgumentException exception) {
            log.error("authorize() failed, authorization code is not in a valid UUID format");
            throw new StateApiException(StateApiErrorCode.E_AUTHORIZATION_CODE_INVALID_FORMAT);
        }

        // Returns authorization-code entity if authorizationCode exists and is valid
        // otherwise exception
        return getAuthorizationCode(authorizationCode).flatMap(ac -> {
            // The state code used to generate the authorization code must match the state
            // code associated with
            // the requester's accountId
            String authorizationCodeStateCode = ac.getStateCode();
            String stateProfileStateCode = saCode.getStateCode();
            if (!stateProfileStateCode.equals(authorizationCodeStateCode)) {
                log.error(
                        "authorize() failed, mismatched state code, state_profile state code : {}, authorization_code state code: {}",
                        stateProfileStateCode,
                        authorizationCodeStateCode);
                throw new StateApiException(StateApiErrorCode.E_MISMATCHED_STATE_CODE);
            }

            // The authorization code must not be expired
            Timestamp expiresAt = ac.getExpiresAt();
            if (expiresAt.getTime() < System.currentTimeMillis()) {
                log.error("authorize() failed, authorization code has expired");
                throw new StateApiException(StateApiErrorCode.E_AUTHORIZATION_CODE_EXPIRED);
            }

            return Mono.just(ac);
        });
    }

    @Override
    public Mono<TaxReturnXml> retrieveTaxReturnXml(int taxYear, UUID taxReturnUuid, String submissionId) {
        log.info("enter retrieveTaxReturnXml()...taxReturnUuid={}", taxReturnUuid);

        return dfClient.getStatus(taxYear, taxReturnUuid, submissionId).flatMap(taxReturnStatus -> {
            if (STATUS_REJECTED.equalsIgnoreCase(taxReturnStatus.status())) {
                log.error("Requested tax return has status of 'rejected'");
                return Mono.error(new StateApiException(StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING));
            }

            if (STATUS_ERROR.equalsIgnoreCase(taxReturnStatus.status())) {
                log.error("Requested tax return has status of 'error'");
                throw new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
            }

            return stateApiS3Client
                    .getTaxReturnXml(taxYear, taxReturnUuid, submissionId)
                    .map(xmlString -> new TaxReturnXml(taxReturnStatus.status().toLowerCase(), submissionId, xmlString))
                    .onErrorMap(e -> {
                        log.error(
                                "getTaxReturnXml() failed for taxYear={}, taxReturnId={}, submissionId={}, {}, error: {}",
                                taxYear,
                                taxReturnUuid.toString(),
                                submissionId,
                                e.getClass().getName(),
                                e.getMessage());
                        return new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                    });
        });
    }

    @Override
    public Mono<Map<String, Object>> retrieveExportedFacts(String submissionId, String stateCode, String accountId) {
        log.info(
                "enter retrieveExportedFacts()...submissionId={}, acctId={}, enabled={}",
                submissionId,
                accountId,
                exported_facts_enabled);
        if (exported_facts_enabled) {
            return efClient.getExportedFacts(submissionId, stateCode, accountId)
                    .map(GetStateExportedFactsResponse::exportedFacts);
        }
        return Mono.just(new HashMap<>());
    }

    @Override
    public Mono<EncryptData> encryptTaxReturn(TaxReturnToExport taxReturn, String accountId) {
        log.info("enter encryptTaxReturn()...accountId={}", accountId);

        // Compared AES 256 CBC, CCM and GCM, we chose GCM for combining data
        // confidentiality and integrity/authentication in a highly efficient manner.
        return getStateProfile(accountId)
                .flatMap(sp -> {
                    return retrievePublicKeyFromCert(sp);
                })
                .map(publicKey -> {
                    try {
                        // 1. generate secret
                        byte[] secret = generatePassword();

                        // 2. generate a random initialization vector for AES CBC
                        byte[] iv = generateIV();

                        // 3. encrypt the xml data with AES 256 GCM
                        var gsonBuilder = new GsonBuilder();
                        gsonBuilder.serializeNulls();
                        Gson gson = gsonBuilder.create();
                        AesGcmEncryptionResult encryptedResult = aesGcmEncrypt(gson.toJson(taxReturn), secret, iv);

                        // 4. Encode the encrypted xml
                        String encodedAndEncryptedData = Base64.toBase64String(encryptedResult.ciphertext());

                        // 5. encrypt the secret with state's public key and encode with base64
                        byte[] encryptedSecret = rsaEncryptWithPublicKey(secret, publicKey);

                        String encodedSecret = Base64.toBase64String(encryptedSecret);

                        // 6. encode iv
                        String encodedIV = Base64.toBase64String(iv);

                        // 7. encode authentication tag
                        String encodedAuthenticationTag = Base64.toBase64String(encryptedResult.authenticationTag());

                        return new EncryptData(
                                encodedSecret, encodedIV, encodedAndEncryptedData, encodedAuthenticationTag);
                    } catch (Exception e) {
                        log.error(
                                "encryptTaxReturnXml() failed, {}, error: {}",
                                e.getClass().getName(),
                                e.getMessage());
                        throw new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                    }
                });
    }

    private Mono<Boolean> getAcceptedOnlyFlag(String stateCode) {
        log.info("enter getAcceptedOnlyFlag...");

        return lookupStateProfile(stateCode)
                .doOnSuccess(s -> log.info("getAcceptedOnlyFlag completes successfully, flag: {}", s.acceptedOnly()))
                .doOnError(er -> log.error("getAcceptedOnlyFlag fails: {}", er.getMessage()))
                .flatMap(sp -> Mono.just(sp.acceptedOnly()));
    }

    @Override
    public Mono<StateProfileDTO> lookupStateProfile(String stateCode) {
        return cachedDS.getStateProfileByStateCode(stateCode).map(dto -> {
            if (dto.archived()) {
                log.error("State {} is archived", stateCode);
                throw new StateApiException(StateApiErrorCode.E_ACCOUNT_ARCHIVED);
            }
            return dto;
        });
    }

    @Override
    public Mono<StateProfile> getStateProfile(String accountId) {
        return cachedDS.getStateProfile(accountId).map(sp -> {
            if (sp.getArchived()) {
                log.error("State {} (account_id={}) is archived", sp.getStateCode(), accountId);
                throw new StateApiException(StateApiErrorCode.E_ACCOUNT_ARCHIVED);
            }
            return sp;
        });
    }

    private Mono<AuthorizationCode> getAuthorizationCode(UUID authorizationCode) {
        log.info("enter getAuthorizationCode()...");

        AuthorizationCode code = new AuthorizationCode();
        code.setAuthorizationCode(authorizationCode);
        return acRepo.getByAuthorizationCode(code.getAuthorizationCode())
                .switchIfEmpty(Mono.defer(() -> {
                    log.error(
                            "getAuthorizationCode() failed, authorization code {} does not exist in authorization_code table",
                            authorizationCode);
                    return Mono.error(new StateApiException(StateApiErrorCode.E_AUTHORIZATION_CODE_NOT_EXIST));
                }))
                .onErrorMap(e -> !(e instanceof StateApiException), e -> {
                    log.error(
                            "getAuthorizationCode() failed, {}, error: {}",
                            e.getClass().getName(),
                            e.getMessage());
                    return new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                });
    }

    // Check if using default cert in lower env
    private Mono<PublicKey> retrievePublicKeyFromCert(StateProfile sp) {
        String certOverride = certProperties.getCertLocationOverride();
        String cert = (StringUtils.isBlank(certOverride)) ? sp.getCertLocation() : certOverride;
        OffsetDateTime expDate = (StringUtils.isBlank(certOverride))
                ? sp.getCertExpirationDate()
                : OffsetDateTime.now().plusYears(1);
        log.info("Use CertOverride={} to retrieve public key.", certOverride);
        return cachedDS.retrievePublicKeyFromCert(cert, expDate);
    }
}
