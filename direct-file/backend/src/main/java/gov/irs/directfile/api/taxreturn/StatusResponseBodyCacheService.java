package gov.irs.directfile.api.taxreturn;

import java.util.Optional;

import gov.irs.directfile.api.taxreturn.dto.StatusResponseBody;

public interface StatusResponseBodyCacheService {
    /**
     * Returns a cached StatusResponseBody for the given submission ID, or an empty
     * Optional if no entry found.
     */
    Optional<StatusResponseBody> get(String submissionId);

    /**
     * Puts the specified StatusResponseBody into the cache for the given submission ID.
     */
    void put(String submissionId, StatusResponseBody statusResponseBody);

    /**
     * Deletes the cache entry for the specified submission ID.
     */
    void clearKey(String submissionId);

    /**
     * Deletes the entire cache of all StatusResponseBody entries (currently only used in tests).
     */
    void clear();
}
