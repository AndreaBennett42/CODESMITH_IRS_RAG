package gov.irs.directfile.stateapi.service;

import java.security.PublicKey;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import gov.irs.directfile.error.StateApiErrorCode;
import gov.irs.directfile.stateapi.dto.StateProfileDTO;
import gov.irs.directfile.stateapi.exception.StateApiException;
import gov.irs.directfile.stateapi.exception.StateNotExistException;
import gov.irs.directfile.stateapi.model.StateLanguage;
import gov.irs.directfile.stateapi.model.StateProfile;
import gov.irs.directfile.stateapi.model.StateRedirect;
import gov.irs.directfile.stateapi.repository.StateApiS3Client;
import gov.irs.directfile.stateapi.repository.StateLanguageRepository;
import gov.irs.directfile.stateapi.repository.StateProfileRepository;
import gov.irs.directfile.stateapi.repository.StateRedirectRepository;

@Component
@Slf4j
@SuppressWarnings("PMD.PreserveStackTrace")
public class CachedDataService {
    @Autowired
    private StateApiS3Client s3Client;

    @Autowired
    private StateProfileRepository spRepo;

    @Autowired
    private StateRedirectRepository srRepo;

    @Autowired
    private StateLanguageRepository slRepo;

    @Value("${spring.cache.TTL-minutes: 120}")
    private long cacheTTL;

    // Note: We are applying cache of cache to a Mono. The native Caffeine cache's 'expireAfterAccess' won't take
    // effect. For the sake of simplicity, we periodically evict the caches.
    @CacheEvict(
            value = {"publicKeyCache", "stateProfileCache"},
            allEntries = true)
    @Scheduled(fixedRateString = "${spring.cache.TTL-minutes}", timeUnit = TimeUnit.MINUTES)
    public void emptyCaches() {
        log.info("caches (publicKeyCache, stateProfileCache) were evicted after {} minutes", cacheTTL);
    }

    // NOTE: the cert is cached and expiration won't apply during the cache duration
    @Cacheable(cacheNames = "publicKeyCache", key = "#certName")
    public Mono<PublicKey> retrievePublicKeyFromCert(String certName, OffsetDateTime enforcedExpirationDate) {
        log.info("enter retrievePublicKeyFromCert()...for {}", certName);

        return s3Client.getCert(certName)
                .flatMap(is -> {
                    X509Certificate cert;
                    try {
                        CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
                        cert = (X509Certificate) certFactory.generateCertificate(is);
                    } catch (CertificateException e) {
                        log.error(
                                "retrievePublicKeyFromCert failed, {}, {}",
                                e.getClass().getName(),
                                e.getMessage());
                        throw new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                    }

                    // Check if the certificate is expired
                    Date currentDate = new Date();
                    if (currentDate.after(cert.getNotAfter())) {
                        log.error("The certificate {} has expired", certName);
                        throw new StateApiException(StateApiErrorCode.E_CERTIFICATE_EXPIRED);
                    }

                    // check IRS enforced expiration date
                    if (enforcedExpirationDate != null) {
                        OffsetDateTime currentDateTime = OffsetDateTime.now(ZoneOffset.UTC);
                        if (currentDateTime.isAfter(enforcedExpirationDate)) {
                            log.error("The certificate {} has passed the IRS enforced expiration date", certName);

                            throw new StateApiException(StateApiErrorCode.E_CERTIFICATE_EXPIRED);
                        }
                    }
                    return Mono.just(cert.getPublicKey());
                })
                .cache(); // This is the hack to make @Cacheable work #https://www.baeldung.com/spring-webflux-cacheable
    }

    @Cacheable(cacheNames = "stateProfileCache", key = "#accountId")
    public Mono<StateProfile> getStateProfile(String accountId) {
        log.info("enter getStateProfile()...accountId={}", accountId);

        return spRepo.getByAccountId(accountId)
                .switchIfEmpty(Mono.defer(() -> {
                    log.error(
                            "getStateProfile() failed, account id does not exist in state_profile table for account id: {}",
                            accountId);
                    return Mono.error(new StateApiException(StateApiErrorCode.E_ACCOUNT_ID_NOT_EXIST));
                }))
                .onErrorMap(e -> !(e instanceof StateApiException), e -> {
                    log.error(
                            "getStateProfile failed for account id: {}, {}, error: {}",
                            accountId,
                            e.getClass().getName(),
                            e.getMessage());
                    return new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                })
                .cache();
    }

    @Cacheable(cacheNames = "stateProfileCache", key = "#stateCode")
    public Mono<StateProfileDTO> getStateProfileByStateCode(String stateCode) {
        log.info("enter getStateProfileByStateCode()...stateCode={}", stateCode);

        return spRepo.getByStateCode(stateCode)
                .flatMap(this::loadRelations)
                .switchIfEmpty(Mono.defer(() -> {
                    log.info("No StateProfile returns, state code {} does not exist in state_profile table", stateCode);
                    return Mono.error(new StateNotExistException(StateApiErrorCode.E_STATE_NOT_EXIST));
                }))
                .onErrorMap(e -> !(e instanceof StateApiException), e -> {
                    log.error(
                            "getStateProfileByStateCode() failed for state code: {}, {}, error: {}",
                            stateCode,
                            e.getClass().getName(),
                            e.getMessage());

                    return new StateApiException(StateApiErrorCode.E_INTERNAL_SERVER_ERROR);
                })
                .cache();
    }

    private Mono<StateProfileDTO> loadRelations(final StateProfile stateProfile) {
        var stateProfileId = stateProfile.getId();

        // Load the redirect urls
        Mono<List<StateRedirect>> redirectUrls =
                srRepo.getAllByStateProfileId(stateProfileId).collectList();
        // Load the languages
        Mono<List<StateLanguage>> stateLanguages =
                slRepo.getAllByStateProfileId(stateProfileId).collectList();

        return redirectUrls
                .zipWith(stateLanguages)
                .map((urlsAndLanguagesTuple) -> new StateProfileDTO(
                        stateProfile, urlsAndLanguagesTuple.getT1(), urlsAndLanguagesTuple.getT2()));
    }
}
