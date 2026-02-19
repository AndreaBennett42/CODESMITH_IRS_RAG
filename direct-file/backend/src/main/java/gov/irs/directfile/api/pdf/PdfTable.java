package gov.irs.directfile.api.pdf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.UUID;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.interactive.form.PDAcroForm;
import org.apache.pdfbox.pdmodel.interactive.form.PDField;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

@Slf4j
class PdfTable extends PdfTemplate {

    public PdfTable(final String templateName) {
        super(templateName, false, false);
    }

    @Override
    public List<PDDocument> generateDocuments(
            final FactEvaluationResult facts,
            final byte[] pdfTemplateBytes,
            final PdfConfiguration pdfConfig,
            final PdfLanguages language)
            throws PdfCreationException {
        final var results = new ArrayList<PDDocument>();

        // See if this table should be included.
        if (!facts.getBoolean(pdfConfig.getIncludeWhenFactPath())) return results;
        final TableConfig tableConfig = pdfConfig.getTableConfig();
        if (tableConfig == null) return results;

        // Get the collection and skip any items already listed elsewhere.
        final Iterator<UUID> rowCollectionIter =
                getDependentsBeyondFirstFew(facts, tableConfig.rowsCollectionPath(), tableConfig.itemsToSkip());

        // Assemble data for the remaining items and populate table page(s).
        int pageNumber = 0;
        int rowNumber = 0;

        @SuppressWarnings("PMD.CloseResource")
        PDDocument newPdfDocument = null;
        PDAcroForm acroForm = null;
        final String prefix = this.getTemplateName() + "_";
        try {
            while (rowCollectionIter.hasNext()) {
                // Each collection item generates a table row.
                final UUID itemId = rowCollectionIter.next();
                ++rowNumber;
                if (rowNumber % tableConfig.rowsPerPage() == 1) {
                    // If we've just populated a template page ...
                    if (newPdfDocument != null) {
                        // ... populate page headers, footers, etc. ...
                        populateOncePerPageFields(
                                facts, tableConfig.oncePerPageFields(), acroForm, prefix + pageNumber, language);
                        // ... and add the page to the results.
                        results.add(newPdfDocument);
                    }
                    // Load the template to create a new page.
                    newPdfDocument = Loader.loadPDF(pdfTemplateBytes);
                    acroForm = newPdfDocument.getDocumentCatalog().getAcroForm();
                    pageNumber++;
                    rowNumber = 1;
                }

                // Populate the row's columns.
                for (final PdfTableFactExpressionMapping column : tableConfig.columns()) {
                    try {
                        final String fieldName = column.fieldName() + rowNumber;
                        Object factValue =
                                determineFieldValue(fieldName, column.factExpression(), facts, language, itemId);
                        if (factValue.getClass() == Boolean.class) {
                            factValue = language.translateBoolean((Boolean) factValue);
                        }
                        if (factValue.toString().length() > 0) {
                            setPdfField(fieldName, factValue.toString(), acroForm, prefix + pageNumber);
                        }
                    } catch (IOException | PdfCreationException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        } catch (IOException e) {
            throw new PdfCreationException("Could not load PDF table template input stream", e);
        }

        // Add the final page to the results.
        if (newPdfDocument != null) {
            try {
                populateOncePerPageFields(
                        facts, tableConfig.oncePerPageFields(), acroForm, prefix + pageNumber, language);
                results.add(newPdfDocument);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return results;
    }

    private void populateOncePerPageFields(
            final FactEvaluationResult facts,
            final List<PdfTableFactExpressionMapping> oncePerPageFields,
            final PDAcroForm acroForm,
            final String prefix,
            final PdfLanguages language)
            throws PdfCreationException, IOException {
        for (final PdfTableFactExpressionMapping oncePerPageField : oncePerPageFields) {
            final Object factValue = determineFieldValue(
                    oncePerPageField.fieldName(), oncePerPageField.factExpression(), facts, language, null);
            if (factValue.toString().length() > 0) {
                setPdfField(oncePerPageField.fieldName(), factValue.toString(), acroForm, prefix);
            }
        }
    }

    private void setPdfField(
            final String fieldName, final String newValue, final PDAcroForm acroForm, final String prefix)
            throws PdfCreationException, IOException {
        final PDField field = acroForm.getField(fieldName);
        if (field == null) {
            throw new RuntimeException("Failed to find field " + fieldName);
        }
        PdfFieldHandler.setFieldInPDF(field, newValue);
        // Rename field to avoid possible collisions with additional pages.
        field.setPartialName(prefix + "_" + field.getPartialName());
    }

    private Iterator<UUID> getDependentsBeyondFirstFew(
            final FactEvaluationResult facts, final String collectionPath, final int depsToSkip) {
        final Optional<Object> collectionOptional = facts.getOptional(collectionPath);
        if (collectionOptional.isEmpty()) {
            log.warn("Dependents collection ({}) missing from evaluated facts", collectionPath);
            return new ArrayList<UUID>().iterator();
        }

        final List<UUID> dependentsList = (List<UUID>) collectionOptional.get();
        final Iterator<UUID> dependentsIter = dependentsList.iterator();
        try {
            for (int i = 0; i < depsToSkip; i++) {
                dependentsIter.next();
            }
            if (!dependentsIter.hasNext())
                log.warn("Dependents collection ({}) has fewer than {} elements", collectionPath, depsToSkip + 1);
        } catch (NoSuchElementException e) {
            log.warn("Dependents collection ({}) has fewer than {} elements", collectionPath, depsToSkip + 1);
        }
        return dependentsIter;
    }
}
