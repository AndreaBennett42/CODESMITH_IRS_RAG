package gov.irs.directfile.stateapi.repository;

import java.io.InputStream;
import java.util.UUID;

import reactor.core.publisher.Mono;

@SuppressWarnings("PMD.UnnecessaryModifier")
public interface StateApiS3Client {
    public Mono<InputStream> getCert(String certUrl);

    public Mono<String> getTaxReturnXml(int taxYear, UUID taxReturnId, String submissionId);
}
