package gov.irs.directfile.submit.domain;

import java.time.Instant;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class DocumentStoreResource {
    String fullLocation;
    String resourceId;
    Instant lastModified;
}
