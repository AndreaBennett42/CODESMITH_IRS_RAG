package gov.irs.directfile.api.authentication;

import java.util.Map;
import java.util.Set;
import java.util.UUID;

public interface PIIService {
    default String fetchAttribute(UUID userExternalId, PIIAttribute attribute) {
        return fetchAttributes(userExternalId, Set.of(attribute)).get(attribute);
    }

    Map<PIIAttribute, String> fetchAttributes(UUID userExternalId, Set<PIIAttribute> attributes);
}
