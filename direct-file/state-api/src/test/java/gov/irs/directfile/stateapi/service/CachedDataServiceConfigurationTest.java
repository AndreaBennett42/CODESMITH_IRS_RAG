package gov.irs.directfile.stateapi.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import gov.irs.directfile.stateapi.model.StateProfile;
import gov.irs.directfile.stateapi.repository.StateLanguageRepository;
import gov.irs.directfile.stateapi.repository.StateProfileRepository;
import gov.irs.directfile.stateapi.repository.StateRedirectRepository;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

// Upgrading to Spring 3.2+ caused an unhandled runtime exception that we did not detect. The following test class
// is capable of reproducing the exception by allowing the entire spring context to load.
// For testing the cached data service, use CachedDataServiceTest, as @SpringBootTests are expensive and should be used
// extremely sparingly.
@SpringBootTest
@ActiveProfiles("test")
public class CachedDataServiceConfigurationTest {

    @MockBean
    private StateProfileRepository stateProfileRepository;

    @MockBean
    private StateRedirectRepository stateRedirectRepository;

    @MockBean
    private StateLanguageRepository stateLanguageRepository;

    @Autowired
    CachedDataService cachedDataService;

    @Test
    public void theCacheInitializesProperly() throws IllegalStateException {
        var stateCode = "FS";
        var stateProfileId = 1L;
        var stateProfile = mock(StateProfile.class);
        when(stateProfile.getId()).thenReturn(stateProfileId);
        when(stateProfileRepository.getByStateCode(stateCode)).thenReturn(Mono.just(stateProfile));
        when(stateLanguageRepository.getAllByStateProfileId(stateProfileId)).thenReturn(Flux.empty());
        when(stateRedirectRepository.getAllByStateProfileId(stateProfileId)).thenReturn(Flux.empty());

        var resultMono = cachedDataService.getStateProfileByStateCode(stateCode);

        StepVerifier.create(resultMono).expectNextCount(1).verifyComplete();
    }
}
