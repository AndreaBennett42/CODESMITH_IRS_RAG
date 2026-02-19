package gov.irs.directfile.api.pdf;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.pdfbox.pdmodel.interactive.form.PDCheckBox;
import org.apache.pdfbox.pdmodel.interactive.form.PDField;
import org.apache.pdfbox.pdmodel.interactive.form.PDTextField;

public class PdfFieldHandler {
    private static final Pattern fontNameRegEx = Pattern.compile("\\/(?<fontname>\\S+)\\s.*");

    /*
    private static void debugPDF(PDField field, String factGraphPath) throws IOException {
        // Just leaving this here in case anyone ever wants to output
        // a pdf with the facts listed on the PDF.  Might be useful...
        if (field instanceof PDCheckBox) {
            PDCheckBox p = (PDCheckBox) field;
            var positiveValue = p.getOnValue();
            p.setValue(positiveValue);
        } else {
            field.setValue(factGraphPath);
        }
    } */

    public static void setFieldInPDF(final PDField field, final Object factGraphValue)
            throws PdfCreationException, IOException {
        if (field == null) {
            throw new PdfCreationException("Cannot set a null PDField");
        } else if (field instanceof PDCheckBox) {
            PDCheckBox p = (PDCheckBox) field;
            if (factGraphValue.getClass() == Boolean.class) {
                boolean val = (boolean) factGraphValue;
                if (val) {
                    var positiveValue = p.getOnValue();
                    p.setValue(positiveValue);
                }
            } else {
                throw new PdfCreationException("Non-boolean field applied to boolean value");
            }
        } else if (field instanceof PDTextField) {
            try {
                field.setValue(factGraphValue.toString().trim());
            } catch (IOException e) {
                // Careful pre-processing of IRS PDF templates should prevent us losing embedded fonts.
                // See README for related details.
                // If a font is lost, we end up here.
                String appearance = ((PDTextField) field).getDefaultAppearance();
                Matcher matcher = fontNameRegEx.matcher(appearance);
                if (matcher.find()) {
                    // Replace the mising font's name with Helvetica, keeping the rest of the appearance settings.
                    appearance = appearance.replaceFirst(matcher.group("fontname"), "Helv");
                } else {
                    // Don't expect to ever be here, but in case we cannot identify the missing font, use the document's
                    // default appearance.
                    appearance = null;
                }
                // Change the appearance ...
                ((PDTextField) field).setDefaultAppearance(appearance);
                // ... and try again.
                field.setValue(factGraphValue.toString().trim());
            }
        } else {
            throw new PdfCreationException(
                    "Unhandled PDField type " + field.getClass().getName());
        }
    }
}
