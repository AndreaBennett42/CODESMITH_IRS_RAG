package gov.irs.directfile.api.pdf.load;

import java.util.HashMap;
import java.util.Map;

import gov.irs.directfile.api.pdf.PdfCreationException;

class ConfiguredPdfMap {
    // by year, then form, then language
    private final Map<String, Map<String, Map<String, ConfiguredPdf>>> configuredPdfs = new HashMap<>();
    private static final String DEFAULT_LANGUAGE = "en";

    private Map<String, Map<String, ConfiguredPdf>> getYear(String year) {
        if (!configuredPdfs.containsKey(year)) {
            configuredPdfs.put(year, new HashMap<>());
        }
        return configuredPdfs.get(year);
    }

    public Map<String, ConfiguredPdf> getFormLanguages(String year, String form) {
        var map = getYear(year);
        if (!map.containsKey(form)) {
            map.put(form, new HashMap<>());
        }
        return map.get(form);
    }

    public ConfiguredPdf getForm(String year, String form, String lang, boolean getDefaultIfNotAvailable)
            throws PdfCreationException {
        // do not want to use creation methods when getting a form
        if (!configuredPdfs.containsKey(year)) {
            throw new PdfCreationException(String.format("Missing year in PDF creation: %s", year));
        }
        var forms = configuredPdfs.get(year);
        if (!forms.containsKey(form)) {
            throw new PdfCreationException(String.format("Missing form %s in year %s for PDF creation", form, year));
        }
        var languages = getFormLanguages(year, form);
        var value = languages.get(lang);
        if (value == null && !DEFAULT_LANGUAGE.equals(lang) && getDefaultIfNotAvailable)
            return languages.get(DEFAULT_LANGUAGE);
        return value;
    }
}
