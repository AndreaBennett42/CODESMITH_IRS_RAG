package gov.irs.directfile.api.dataimport.gating;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.config.DataImportGatingConfigurationProperties;
import gov.irs.directfile.api.config.DataImportGatingConfigurationProperties.Allowlist;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class DataImportGatingEmailAllowlistServiceTest {

    @Mock
    DataImportGatingConfigService dataImportGatingConfigService;

    private DataImportGatingEmailAllowlistService emailAllowlistService;

    private String hexKey = "c6b27cc233024f50dd90a826dc7ae79936c29a599791b14cd0eb0e48e1d5cfff";

    // test@example.com
    // example2@example.com
    // EXAMPLE3@EXAMPLE.COM
    private String allowListExportCsv =
            """
			9bO/RaqAl1I4aeexSsadrHOkxKfiWNhpItXFc5KmIrs=
			Zq5rD40EW55DnI35KYdK7f+u16lblwn+8H3YUdrxWsw=
			9yO0gpET8JXfw45WC84bm4K2x7lXUo+CVmgTcv7/KzU=
			""";

    @BeforeEach
    void setup() {
        Allowlist allowlist = new Allowlist(true, hexKey, "data-import-allowlist-export.csv");
        DataImportGatingConfigurationProperties props = new DataImportGatingConfigurationProperties(allowlist);
        emailAllowlistService = new DataImportGatingEmailAllowlistService(props, dataImportGatingConfigService);
    }

    @Test
    public void testEmailOnAllowlist_contains_thenReturnsTrue() {
        when(dataImportGatingConfigService.getDataImportGatingObjectAsString(any()))
                .thenReturn(allowListExportCsv);

        boolean exists = emailAllowlistService.emailOnAllowlist("test@example.com");

        assertTrue(exists);
    }

    @Test
    public void testEmailOnAllowlist_notContains_thenReturnsFalse() {
        when(dataImportGatingConfigService.getDataImportGatingObjectAsString(any()))
                .thenReturn(allowListExportCsv);

        boolean exists = emailAllowlistService.emailOnAllowlist("xxx@example.com");

        assertFalse(exists);
    }

    @Test
    public void testEmailOnAllowlist_exception_thenReturnsFalse() {
        when(dataImportGatingConfigService.getDataImportGatingObjectAsString(any()))
                .thenThrow(new RuntimeException());

        boolean exists = emailAllowlistService.emailOnAllowlist("test@example.com");

        assertFalse(exists);
    }
}
