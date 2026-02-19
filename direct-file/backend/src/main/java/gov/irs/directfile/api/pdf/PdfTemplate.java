package gov.irs.directfile.api.pdf;

import java.math.RoundingMode;
import java.text.NumberFormat;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

@Slf4j
abstract class PdfTemplate {
    protected static final String newLine = System.getProperty("line.separator");
    protected static final Pattern factPathRegEx = Pattern.compile("[/~]\\S+");
    protected String templateName;
    protected boolean shouldRemoveTinDashes;
    protected boolean shouldRoundCurrency;
    private static NumberFormat currencyRounder = null;

    // Order by Attachment Sequence Number (ASN) in upper right of IRS forms.
    protected static List<PdfTemplate> templateList = List.of(
            new Irs1040PdfForm("IRS1040"),
            new Irs1040PdfForm("IRS1040SR"),
            new PdfTable("DEPENDENTS_STATEMENT"),

            // ASN 01
            new PdfForm("IRS1040S1"),

            // ASN 02
            new PdfForm("IRS1040S2"),

            // ASN 03
            new PdfForm("IRS1040S3"),

            // ASN 08
            new PdfForm("IRS1040SB"),

            // ASN 16
            new PdfForm("SCHEDULE_R"),

            // ASN 21
            new Irs2441PdfForm("IRS2441"),
            new PdfTable("CDCC_QUALIFYING_PERSON_STATEMENT"),
            new PdfTable("CDCC_CAREGIVERS_STATEMENT"),
            new Irs2441DueDiligencePdfTable("IRS2441_DUE_DILIGENCE_STATEMENT"),

            // ASN 43
            new IrsScheduleEicPdfForm("IRS1040EIC"),

            // (in TY23, Form 8862 was here with ASN 43A)

            // ASN 47
            new PdfForm("IRS1040S8812"),

            // ASN 52
            new PdfForm("IRS8889"),

            // ASN 54
            new PdfForm("IRS8880"),

            // ASN 73
            new PdfForm("IRS8962"),

            // ASN 77
            new Irs9000PdfForm("FORM9000"),

            // ASN 77A
            new IrsScheduleLepPdfForm("SCHEDULE_LEP"),

            // ASN 862
            new Irs8862PdfForm("IRS8862"),
            new PdfTable("CTC_DEPENDENTS_STATEMENT"),
            new PdfTable("ODC_DEPENDENTS_STATEMENT"),

            // no ASN
            new IrsW2PdfForm("IRSW2"),

            // no ASN
            new PdfForm("DF1099R"));

    protected PdfTemplate(
            final String templateName, final boolean shouldRemoveTinDashes, final boolean shouldRoundCurrency) {
        this.templateName = templateName;
        this.shouldRemoveTinDashes = shouldRemoveTinDashes;
        this.shouldRoundCurrency = shouldRoundCurrency;
    }

    protected String getTemplateName() {
        return this.templateName;
    }

    protected abstract List<PDDocument> generateDocuments(
            final FactEvaluationResult facts,
            final byte[] pdfTemplateBytes,
            final PdfConfiguration pdfConfig,
            final PdfLanguages language)
            throws PdfCreationException;

    protected Object determineFieldValue(
            final String pdfFieldName,
            final String factExpression,
            final FactEvaluationResult facts,
            final PdfLanguages language,
            final UUID collectionItemId) {
        // A fact expression is a sequence of tokens. A token can be a fact path, `space`, or `newLine`.
        // Token values are concatenated from left to right.
        final String[] tokens = factExpression.split("\\s+");
        StringBuilder fieldValue = new StringBuilder("");

        // When there is one token, handle it directly. Value could be String or Boolean.
        if (tokens.length == 1) {
            return evaluateToken(tokens[0], fieldValue.toString(), pdfFieldName, facts, language, collectionItemId);
        }

        // When there are multiple tokens, concatenate values from left to right.
        // All values should be String; it doesn't really make sense to concatenate a Boolean.
        for (final String token : tokens) {
            final Object tokenValue =
                    evaluateToken(token, fieldValue.toString(), pdfFieldName, facts, language, collectionItemId);
            if (tokenValue.getClass() == Boolean.class) {
                log.warn(
                        "Ignoring Boolean in compound fact expression for template {} field {}",
                        this.templateName,
                        pdfFieldName);
                continue;
            }
            fieldValue.append(tokenValue.toString());
        }

        return fieldValue.toString().trim();
    }

    protected Object evaluateToken(
            final String token,
            final String fieldValue,
            final String pdfFieldName,
            final FactEvaluationResult facts,
            final PdfLanguages language,
            final UUID collectionItemId) {

        // This function must return a non-null object, Boolean or String.
        // If nothing else, return:
        final String blank = "";

        if ("space".equals(token)) {
            return " ";
        }

        if ("newLine".equals(token)) {
            // Avoid sequential newLines due to previous blank token(s).
            if (!fieldValue.endsWith(newLine)) return newLine;
            return blank;
        }

        if (!factPathRegEx.matcher(token.trim()).matches()) {
            log.warn(
                    "Unknown token {} in fact expression for template {} field {}",
                    token,
                    this.templateName,
                    pdfFieldName);
            return blank;
        }

        String factPath = token.trim();

        // When in the context of a collection item, inject the ID into the path.
        if (collectionItemId != null) {
            factPath = factPath.replace("*", "#" + collectionItemId.toString());
        }

        final Optional<Object> opt = facts.getOptional(factPath);
        if (!opt.isPresent()) return blank;
        final Object tokenValue = opt.get();
        final Object tokenClass = tokenValue.getClass();

        if (tokenClass == Boolean.class) {
            return tokenValue;
        }

        final String tokenValueAsString = tokenValue.toString().trim();

        if (textValueClasses.contains(tokenClass)) {
            return tokenValueAsString;
        }

        if (tokenClass == gov.irs.factgraph.types.UsPhoneNumber.class) {
            if (tokenValueAsString.startsWith("+1")) {
                return tokenValueAsString.substring(2);
            }
            return tokenValueAsString;
        }

        if (tokenClass == gov.irs.factgraph.types.Tin.class) {
            if (this.shouldRemoveTinDashes) {
                return tokenValueAsString.replace("-", "");
            }
            return tokenValueAsString;
        }

        if (tokenClass == scala.math.BigDecimal.class) {
            if (this.shouldRoundCurrency) {
                return roundCurrency(tokenValue.toString());
            }
            return tokenValue.toString();
        }

        if (tokenClass != gov.irs.factgraph.types.Enum.class) {
            log.warn("Unexpected class type {}", tokenClass);
            return blank;
        }

        final gov.irs.factgraph.types.Enum enumObject = (gov.irs.factgraph.types.Enum) tokenValue;
        if (enumObject.value().isEmpty()) {
            return blank;
        }

        final String enumValue = enumObject.value().get();
        final String optionsType = enumObject.enumOptionsPath();

        if ("/scopedStateOptions".equals(optionsType)) {
            return enumValue.toUpperCase();
        }

        if ("/relationshipOptions".equals(optionsType)) {
            return language.translateRelationship(enumValue);
        }

        return enumValue;
    }

    // Assumes arg can be parsed as a Double which will be rounded and formatted
    protected String roundCurrency(String amount) {
        try {
            final String r = getCurrencyRounder().format(Double.parseDouble(amount));
            return r;
        } catch (NumberFormatException numberFormatException) {
            return amount;
        }
    }

    public static NumberFormat getCurrencyRounder() {
        if (currencyRounder == null) {
            currencyRounder = NumberFormat.getInstance(Locale.US);
            currencyRounder.setMaximumFractionDigits(0);
            currencyRounder.setRoundingMode(RoundingMode.HALF_EVEN);
        }
        return currencyRounder;
    }

    protected static final Set<Class<?>> textValueClasses = Set.of(
            String.class,
            Integer.class,
            gov.irs.factgraph.types.Ein.class,
            gov.irs.factgraph.types.Address.class,
            gov.irs.factgraph.types.EmailAddress.class,
            gov.irs.factgraph.types.Pin.class,
            gov.irs.factgraph.types.IpPin.class);
}
