package gov.irs.directfile.stateapi.repository;

import java.util.UUID;

import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.model.TaxReturnStatus;

@SuppressWarnings("PMD.UnnecessaryModifier")
public interface DirectFileClient {
    static String STATUS_ACCEPTED = "accepted";
    static String STATUS_REJECTED = "rejected";
    static String STATUS_PENDING = "pending";
    static String STATUS_ERROR = "error";

    Mono<TaxReturnStatus> getStatus(int taxYear, UUID taxReturnId, String submissionId);
}
