package gov.irs.directfile.api.pdf.load;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;

import gov.irs.directfile.api.config.ConfiguredPdfProperties;
import gov.irs.directfile.api.errors.DefaultCaseException;
import gov.irs.directfile.api.io.IOLocationException;
import gov.irs.directfile.api.io.IOLocationService;
import gov.irs.directfile.api.pdf.PdfCreationException;

@Slf4j
@SuppressFBWarnings(value = "CT_CONSTRUCTOR_THROW", justification = "Java 21 update")
public class ConfiguredPdfLookup {
    private static final String locationPrefix = "pdflookup";

    private final ConfiguredPdfMap map = new ConfiguredPdfMap();

    private String normalize(String value) {
        return value.toLowerCase().trim();
    }

    public ConfiguredPdfLookup(IOLocationService ioLocationService, List<ConfiguredPdfProperties> properties)
            throws IOLocationException, IOException, DefaultCaseException {
        for (int i = 0; i < properties.size(); i++) {
            var prop = properties.get(i);
            var name = normalize(prop.getName());
            var year = prop.getYear();
            var lang = normalize(prop.getLanguageCode());
            if (lang.length() > 2) throw new RuntimeException("PDF generation accepts only 2 letter language codes");
            log.info("Getting stream access to PDF");
            AtomicReference<InputStream> pdfStream = new AtomicReference<>(ioLocationService.read(
                    IOLocationService.getConfiguredLocationType(prop.getLocationType()), prop.getLocation()));
            log.info("Getting stream access to PDF config");
            AtomicReference<InputStream> configStream = new AtomicReference<>(ioLocationService.read(
                    IOLocationService.getConfiguredLocationType(prop.getConfigurationLocationType()),
                    prop.getConfigurationLocation()));
            // if we need to, put the PDF in memory to make it work faster!
            if (prop.isCacheInMemory()) {
                ioLocationService.write(
                        IOLocationService.ConfiguredLocations.memory,
                        inMemoryCacheName(name, year, lang, false),
                        pdfStream.get());
                pdfStream.set(ioLocationService.read(
                        IOLocationService.ConfiguredLocations.memory, inMemoryCacheName(name, year, lang, false)));
            }
            // the configs are small, so they should be in memory generally
            ioLocationService.write(
                    IOLocationService.ConfiguredLocations.memory,
                    inMemoryCacheName(name, year, lang, true),
                    configStream.get());
            configStream.set(ioLocationService.read(
                    IOLocationService.ConfiguredLocations.memory, inMemoryCacheName(name, year, lang, true)));
            var languages = map.getFormLanguages(year, name);
            var configuredPdf = new ConfiguredPdf(
                    pdfStream.get(), PdfConfiguration.load(configStream.get()), prop.getPagesToInclude());
            languages.put(lang, configuredPdf);
        }
    }

    private String inMemoryCacheName(String name, String year, String language, boolean config) {
        if (!config) {
            return String.format("%s:%s-%s-%s", locationPrefix, year, name, language);
        } else {
            return String.format("%s:%s-%s-%s-config", locationPrefix, year, name, language);
        }
    }

    public ConfiguredPdf getBlankPdfStream(String year, String formName, String language) throws PdfCreationException {
        // if the form isn't available in the specified language, it may still
        // be available in english.  This should probably be configurable
        var stream = map.getForm(year, normalize(formName), normalize(language), true);
        if (stream == null) throw new PdfCreationException(String.format("%s form missing", formName));

        try {
            // Not the ideal solution, but solves the error from multiple reads for a file stored in the map
            // Problem: putting the ConfiguredPdf blankPDF stream into the map then reading it moves the position
            // to the end of the file causing an unexpected error on the second read of that stream
            // (e.g. two calls to generate a PDF )
            stream.reset();
        } catch (IOException e) {
            throw new PdfCreationException(e);
        }
        return stream;
    }
}
