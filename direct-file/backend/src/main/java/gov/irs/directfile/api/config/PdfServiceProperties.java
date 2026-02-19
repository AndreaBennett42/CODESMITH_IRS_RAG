package gov.irs.directfile.api.config;

import java.util.List;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@AllArgsConstructor
@ConfigurationProperties(prefix = "direct-file.pdfs")
@SuppressFBWarnings(value = "EI_EXPOSE_REP", justification = "Initial Spotbugs Setup")
public class PdfServiceProperties {
    private List<ConfiguredPdfProperties> configuredPdfs;
    private String outputLocation;
    private String outputLocationType;
    private boolean useDocumentStorageForPilotYear;
}
