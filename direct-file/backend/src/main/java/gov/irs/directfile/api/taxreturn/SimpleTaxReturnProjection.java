package gov.irs.directfile.api.taxreturn;

import java.util.Date;
import java.util.UUID;

/**
 * A Projection is a subset of fields on an entity that allows JPA
 * to only query for the specified fields instead of the entire entity.
 *
 * In this case, the SimpleTaxReturnProjection returns most pertinent columns
 * except for facts and return headers. This allows READ operations to be faster
 * because they don't need to a) pull in large fact graph/return headers and b) make KMS calls.
 *
 * Reference Docs:
 * https://docs.spring.io/spring-data/jpa/reference/data-commons/repositories/projections.html
 * */
public interface SimpleTaxReturnProjection {
    UUID getId();

    Date getCreatedAt();

    Date getUpdatedAt();

    Date getSubmitTime();

    int getTaxYear();
}
