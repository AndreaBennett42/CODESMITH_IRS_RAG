package gov.irs.directfile.api.pdf;

import java.util.Optional;
import java.util.UUID;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

class IrsScheduleLepPdfForm extends PdfForm {

    public IrsScheduleLepPdfForm(final String templateName) {
        super(templateName);
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        final Optional<Object> factOptional = facts.getOptional("/languagePreference");
        // Without a value, the form would be completely blank; return false to avoid merging it.
        if (factOptional.isEmpty()) return false;

        final String value = factOptional.get().toString();
        facts.put("~" + value + "IsSelectedLep", true);
        return true;
    }
}
