package gov.irs.directfile.api.dataimport;

import java.util.Date;
import java.util.UUID;

import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;

public interface DataImportService {
    void sendPreFetchRequest(UUID taxReturnId, UUID userId, UUID externalId, String tin, int taxYear);

    WrappedPopulatedData getPopulatedData(UUID taxReturnId, UUID userId, Date taxReturnCreatedAt);
}
