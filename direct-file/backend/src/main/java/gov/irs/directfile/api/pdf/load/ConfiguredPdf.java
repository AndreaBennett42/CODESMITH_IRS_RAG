package gov.irs.directfile.api.pdf.load;

import java.io.IOException;
import java.io.InputStream;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class ConfiguredPdf {
    private InputStream blankPDF;
    private PdfConfiguration config;
    private int[] pagesToInclude;

    public void reset() throws IOException {
        blankPDF.reset();
    }
}
