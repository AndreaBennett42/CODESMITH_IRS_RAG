package gov.irs.directfile.api.io.documentstore;

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
