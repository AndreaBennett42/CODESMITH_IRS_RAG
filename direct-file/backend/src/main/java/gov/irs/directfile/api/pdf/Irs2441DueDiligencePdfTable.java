package gov.irs.directfile.api.pdf;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.pdfbox.pdmodel.PDDocument;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

public class Irs2441DueDiligencePdfTable extends PdfTable {
    static final String CDCC_CARE_PROVIDERS_WITH_DUE_DILIGENCE_REASONS = "/cdccCareProvidersWithDueDiligenceReasons";
    static final String CDCC_CARE_PROVIDERS = "/cdccCareProviders";
    static final String PDF_DUE_DILIGENCE_PSEUDO_PATH = "~dueDiligence";

    private static final String CANNOT_FIND_REASON_KEY = "providerMovedAndFilerUnableToFindThem";
    private static final String REFUSED_REASON_KEY = "providerRefusedToProvideTIN";
    private static final String CANNOT_FIND_REASON_VALUE =
            "THE PROVIDER HAS MOVED AND I AM UNABLE TO FIND THE PROVIDER TO GET THE TIN";
    private static final String REFUSED_REASON_VALUE = "THE PROVIDER HAS REFUSED TO GIVE ME THE TIN";
    private static final String PATH_FORMAT = "%s/#%s/%s";

    public Irs2441DueDiligencePdfTable(final String templateName) {
        super(templateName);
    }

    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        final Optional<Object> careProvidersWithDueDiligenceReasons =
                facts.getOptional(CDCC_CARE_PROVIDERS_WITH_DUE_DILIGENCE_REASONS);

        if (careProvidersWithDueDiligenceReasons.isPresent()) {
            final List<UUID> careProviderIds = (List<UUID>) careProvidersWithDueDiligenceReasons.get();
            for (final UUID careProviderId : careProviderIds) {
                var dueDiligenceReasonsString = facts.getString(
                        String.format(PATH_FORMAT, CDCC_CARE_PROVIDERS, careProviderId.toString(), "dueDiligence"));

                Pattern pattern = Pattern.compile("Set\\((.*?)\\)");
                Matcher reasons = pattern.matcher(dueDiligenceReasonsString);

                var psuedoReasonPath =
                        String.format(PATH_FORMAT, CDCC_CARE_PROVIDERS, careProviderId, PDF_DUE_DILIGENCE_PSEUDO_PATH);

                if (reasons.find()) {
                    var pdfReasons = new StringBuilder();
                    String[] reasonsList = reasons.group(1).split(", ");
                    for (String reasonKey : reasonsList) {
                        if (CANNOT_FIND_REASON_KEY.equals(reasonKey)) {
                            pdfReasons.append(CANNOT_FIND_REASON_VALUE);
                        } else if (REFUSED_REASON_KEY.equals(reasonKey)) {
                            pdfReasons.append(REFUSED_REASON_VALUE);
                        } else {
                            throw new Error(String.format("Did not recognize reason key '%s'", reasonKey));
                        }
                    }

                    facts.put(psuedoReasonPath, pdfReasons.toString());
                }
            }
        }

        return true;
    }

    @Override
    public List<PDDocument> generateDocuments(
            final FactEvaluationResult facts,
            final byte[] pdfTemplateBytes,
            final PdfConfiguration pdfConfig,
            final PdfLanguages language)
            throws PdfCreationException {

        this.computePseudoFacts(facts, null, pdfConfig);

        return super.generateDocuments(facts, pdfTemplateBytes, pdfConfig, language);
    }
}
