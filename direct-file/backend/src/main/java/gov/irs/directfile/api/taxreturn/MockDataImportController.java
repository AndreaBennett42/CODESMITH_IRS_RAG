package gov.irs.directfile.api.taxreturn;

import java.util.UUID;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RestController;

import gov.irs.directfile.api.audit.Auditable;
import gov.irs.directfile.api.dataimport.DataImportService;
import gov.irs.directfile.api.dataimport.MockDataImportService;
import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;
import gov.irs.directfile.api.events.EventId;
import gov.irs.directfile.api.pdf.PdfService;
import gov.irs.directfile.api.user.UserService;

@Slf4j
@RestController
class MockDataImportController extends TaxReturnController {

    private MockDataImportService mockDataImportService;

    @Autowired
    private HttpServletRequest request;

    public MockDataImportController(
            TaxReturnService taxReturnService,
            UserService userService,
            PdfService pdfService,
            EncryptionCacheWarmingService cacheWarmingService,
            DataImportService dataImportService) {
        super(taxReturnService, userService, pdfService, cacheWarmingService, dataImportService);
        mockDataImportService = (MockDataImportService) dataImportService;
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_GET_POPULATED_DATA)
    public WrappedPopulatedData getPopulatedData(UUID id) {

        return mockDataImportService.getPopulatedData(
                request.getHeader("x-data-import-profile"), request.getHeader("x-data-import-dob"));
    }
}
