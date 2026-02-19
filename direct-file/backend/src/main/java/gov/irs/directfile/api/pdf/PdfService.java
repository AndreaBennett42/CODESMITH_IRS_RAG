package gov.irs.directfile.api.pdf;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.JsonProcessingException;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.multipdf.PDFMergerUtility;
import org.apache.pdfbox.multipdf.PDFMergerUtility.AcroFormMergeMode;
import org.apache.pdfbox.multipdf.PDFMergerUtility.DocumentMergeMode;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.pdmodel.interactive.form.PDAcroForm;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.factgraph.Graph;

import gov.irs.directfile.api.config.PdfServiceProperties;
import gov.irs.directfile.api.io.IOLocationException;
import gov.irs.directfile.api.io.IOLocationService;
import gov.irs.directfile.api.io.documentstore.DocumentNotFoundException;
import gov.irs.directfile.api.io.storagelocations.StorageLocationBuilder;
import gov.irs.directfile.api.loaders.errors.FactGraphSaveException;
import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.pdf.load.ConfiguredPdfLookup;
import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.models.FactEvaluationResult;

@Service
@Slf4j
@EnableConfigurationProperties(PdfServiceProperties.class)
@SuppressFBWarnings(
        value = {"NP_NULL_ON_SOME_PATH_MIGHT_BE_INFEASIBLE"},
        justification = "Initial Spotbugs setup")
@SuppressWarnings(
        value = {
            "PMD.AssignmentInOperand",
            "PMD.AvoidDuplicateLiterals",
            "PMD.CloseResource",
            "PMD.ExceptionAsFlowControl"
        })
public class PdfService {
    private final PdfServiceProperties properties;
    private final IOLocationService ioLocationService;
    private final FactGraphService factGraphService;
    private final ConfiguredPdfLookup lookup;
    private static final int DIRECT_FILE_PILOT_YEAR = 2023;

    @SneakyThrows
    public PdfService(
            PdfServiceProperties properties, IOLocationService ioLocationService, FactGraphService factGraphService) {
        this.properties = properties;
        this.ioLocationService = ioLocationService;
        this.factGraphService = factGraphService;
        lookup = new ConfiguredPdfLookup(ioLocationService, properties.getConfiguredPdfs());
    }

    public InputStream getTaxReturn(String languageString, TaxReturn taxReturn, boolean useStorage)
            throws PdfCreationException {
        int year = taxReturn.getTaxYear();
        final String baseName = "taxreturn";
        PdfLanguages language = PdfLanguages.fromString(languageString);

        boolean useDocumentStorageForPilotYear =
                taxReturn.getTaxYear() == DIRECT_FILE_PILOT_YEAR && properties.isUseDocumentStorageForPilotYear();

        if (useDocumentStorageForPilotYear) {
            InputStream storageStream = getTaxReturnFromStorage(year, taxReturn.getId(), baseName, language);
            if (storageStream != null) {
                return storageStream;
            } else {
                log.error(
                        "Unable to generate pdf for pilot year tax return: {}. No PDF Found in document storage}",
                        taxReturn.getId());
                throw new PdfCreationException(String.format(
                        "Could not get PDF for tax return %s. No PDF found in document storage", taxReturn.getId()));
            }
        }

        // Optionally, look for the combined tax return PDF in storage.
        if (useStorage) {
            InputStream storageStream = getTaxReturnFromStorage(year, taxReturn.getId(), baseName, language);
            if (storageStream != null) {
                return storageStream;
            }
        }

        // We didn't read it, so to generate it build a set of all the fact paths we might need.
        Set<String> factPaths = new HashSet<String>();
        for (final PdfTemplate template : PdfTemplate.templateList) {
            factPaths.addAll(lookup
                    .getBlankPdfStream(String.valueOf(year), template.getTemplateName(), language.getCode())
                    .getConfig()
                    .getFactPathsForPdf()
                    .stream()
                    .filter(path -> !FactEvaluationResult.isPseudoPath(path))
                    .map(path -> FactEvaluationResult.collectionIndexToWildcard(path))
                    .collect(Collectors.toSet()));
        }

        // Extract those facts.
        FactEvaluationResult facts;
        try {
            Graph graph = factGraphService.getGraph(taxReturn.getFacts());
            facts = factGraphService.extractFacts(factPaths, graph);
        } catch (JsonProcessingException e) {
            throw new PdfCreationException("Could not parse JSON for tax return " + taxReturn.getId(), e);
        } catch (FactGraphSaveException e) {
            throw new PdfCreationException(
                    String.format("Could not save factgraph for tax return %s", taxReturn.getId()), e);
        }

        // Initialize a PDF to hold all the forms and tables.
        PDDocument combinedDocument = new PDDocument();
        combinedDocument.getDocumentCatalog().setAcroForm(new PDAcroForm(combinedDocument));
        combinedDocument.getDocumentCatalog().setLanguage(language.getPdfLanguage());
        PDDocumentInformation info = new PDDocumentInformation();
        info.setSubject("U.S. Individual Income Tax Return" + language.getPdfSubjectSuffix());
        info.setTitle("Direct File download" + language.getPdfTitleSuffix());
        info.setCreator("Direct File");
        info.setModificationDate(Calendar.getInstance());
        combinedDocument.setDocumentInformation(info);

        // Generate the forms and tables, as needed.
        final var subDocuments = new ArrayList<PDDocument>();
        for (final PdfTemplate template : PdfTemplate.templateList) {
            final PdfConfiguration pdfConfig = lookup.getBlankPdfStream(
                            String.valueOf(year), template.getTemplateName(), language.getCode())
                    .getConfig();
            final var newTemplatePdf = lookup.getBlankPdfStream(
                            String.valueOf(year), template.getTemplateName(), language.getCode())
                    .getBlankPDF();
            try {
                subDocuments.addAll(
                        template.generateDocuments(facts, newTemplatePdf.readAllBytes(), pdfConfig, language));
            } catch (IOException e) {
                throw new PdfCreationException(
                        String.format("Could not read PDF template bytes for tax return %s", taxReturn.getId()), e);
            }
        }

        // Finalize the combined PDF of the entire tax return.
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        try {
            mergeDocuments(subDocuments, combinedDocument);
            combinedDocument.save(outputStream);
            combinedDocument.close();
            if (combinedDocument.getNumberOfPages() == 0) {
                throw new PdfCreationException("Insufficient data to generate PDF");
            }
        } catch (IOException e) {
            safelyClosePDDocuments(List.of(combinedDocument));
            safelyClosePDDocuments(subDocuments);
            throw new PdfCreationException("Could not save/close combined tax return document", e);
        }

        ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());

        // Optionally, put the combined tax return PDF in storage.
        if (useStorage) {
            putTaxReturnInStorage(year, taxReturn.getId(), baseName, language, inputStream);
        }

        return inputStream;
    }

    private void mergeDocuments(final List<PDDocument> newDocuments, final PDDocument combinedDocument)
            throws PdfCreationException {
        try {
            final PDFMergerUtility merger = new PDFMergerUtility();
            // Legacy mode renames fields if names collide.
            merger.setDocumentMergeMode(DocumentMergeMode.PDFBOX_LEGACY_MODE);
            merger.setAcroFormMergeMode(AcroFormMergeMode.PDFBOX_LEGACY_MODE);
            for (final PDDocument newDocument : newDocuments) {
                merger.appendDocument(combinedDocument, newDocument);
                newDocument.close();
            }
        } catch (IOException e) {
            throw new PdfCreationException("Could not load input stream", e);
        }
    }

    // Helper to avoid memory leak from unclosed document.
    // Must not throw because it is called from catch blocks.
    private void safelyClosePDDocuments(List<PDDocument> documents) {
        for (final PDDocument document : documents) {
            try {
                document.close();
            } catch (Exception e) {
                log.warn("While handling earlier exception, failed to close PDDocument: {}", e.getMessage());
            }
        }
    }

    private ByteArrayInputStream getTaxReturnFromStorage(
            final int year, final UUID taxReturnId, final String baseName, final PdfLanguages language) {
        final String location =
                StorageLocationBuilder.getTaxReturnDocumentLocation(year, taxReturnId, baseName, language.getCode());
        // If we find the PDF in the document store, return the stream.
        ByteArrayInputStream result = null;

        try (InputStream documentStream =
                ioLocationService.read(IOLocationService.ConfiguredLocations.documentstore, location); ) {
            result = new ByteArrayInputStream(documentStream.readAllBytes());
        } catch (DocumentNotFoundException e) {
            result = null;
        } catch (IOLocationException | IOException e) {
            log.error("Error reading tax return {} from storage: {}", taxReturnId, e.getMessage());
        }

        return result;
    }

    private void putTaxReturnInStorage(
            final int year,
            final UUID taxReturnId,
            final String baseName,
            final PdfLanguages language,
            final InputStream stream)
            throws PdfCreationException {
        final String location =
                StorageLocationBuilder.getTaxReturnDocumentLocation(year, taxReturnId, baseName, language.getCode());
        try {
            ioLocationService.write(IOLocationService.ConfiguredLocations.documentstore, location, stream);
            stream.reset();
        } catch (IOLocationException e) {
            throw new PdfCreationException("Could not save combined tax return document", e);
        } catch (IOException e) {
            throw new PdfCreationException("Could not reset stream after saving combined tax return document", e);
        }
    }
}
