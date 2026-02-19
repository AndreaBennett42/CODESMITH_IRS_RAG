package gov.irs.directfile.api.pdf;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

class IrsW2PdfForm extends PdfForm {

    public IrsW2PdfForm(final String templateName) {
        super(templateName);
        this.shouldRoundCurrency = true;
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        // ... handle Box 12 special case: the PDF has four pairs of code, amount fields.
        // Config maps these to pseudo paths ^\s*\/formW2s\/\*\/~box12(Code|Amount)[1-4]\s*$
        // Here we find up to four Box 12 facts with non-zero amounts, and generate
        // entries with those pseudo paths ...
        final String id = itemId.toString();
        int counter = 1;
        final Map<String, String> box12Data = pdfConfig.getCustomData().get("box12");
        for (final Map.Entry<String, String> box12DataEntry : box12Data.entrySet()) {
            final String path = box12DataEntry.getKey().replace("*", "#" + id);
            final Optional<Object> box12FactOptional = facts.getOptional(path);
            if (box12FactOptional.isPresent()) {
                facts.put("/formW2s/#" + id + "/~box12Amount" + counter, box12FactOptional.get());
                facts.put("/formW2s/#" + id + "/~box12Code" + counter, box12DataEntry.getValue());
                ++counter;
                if (counter > 4) break;
            }
        }

        // ... handle Box 14 special case: the PDF has one field for zero or more code, amount pairs.
        // Config maps this to pseudo path /~box14Text
        // Here we find all Box 14 facts with non-zero amounts, and concatenate an entry for that pseudo path ...
        String box14Text = "";
        final Map<String, String> box14Data = pdfConfig.getCustomData().get("box14");
        for (final Map.Entry<String, String> box14DataEntry : box14Data.entrySet()) {
            final String path = box14DataEntry.getKey().replace("*", "#" + id);
            final Optional<Object> box14FactOptional = facts.getOptional(path);
            if (box14FactOptional.isPresent()) {
                box14Text = box14Text + box14DataEntry.getValue() + ": "
                        + this.roundCurrency(box14FactOptional.get().toString()) + newLine;
            }
        }
        facts.put("/formW2s/#" + id + "/~box14Text", box14Text);
        return true;
    }
}
