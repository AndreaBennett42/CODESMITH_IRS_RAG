package gov.irs.directfile.api.pdf;

import java.util.Optional;
import java.util.UUID;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

class Irs9000PdfForm extends PdfForm {

    public Irs9000PdfForm(final String templateName) {
        super(templateName);
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        final Optional<Object> factOptional = facts.getOptional("/commsFormat");
        // Without a value, the form would be completely blank; return false to avoid merging it.
        if (factOptional.isEmpty()) return false;

        final String value = factOptional.get().toString();
        final CommunicationsFormat communicationsFormat = CommunicationsFormat.valueOf(value);
        facts.put(communicationsFormat.getFactPath(), true);
        return true;
    }

    private enum CommunicationsFormat {
        standardPrint("~pdfForm9000StandardPrintIsSelected"),
        largePrint("~pdfForm9000LargePrintIsSelected"),
        braille("~pdfForm9000BrailleIsSelected"),
        audio("~pdfForm9000AudioIsSelected"),
        txt("~pdfForm9000PlainTextFileIsSelected"),
        brf("~pdfForm9000BrailleReadyFileIsSelected");

        private String factPath;

        public String getFactPath() {
            return this.factPath;
        }

        CommunicationsFormat(String factPath) {
            this.factPath = factPath;
        }
    }
}
