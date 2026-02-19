package gov.irs.directfile.api.pdf;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

class IrsScheduleEicPdfForm extends PdfForm {
    public IrsScheduleEicPdfForm(final String templateName) {
        super(templateName);
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        final String EITC_DEPENDENTS = "/scheduleEicDependents";

        final String[] pathSuffixes = {"Millennium", "Century", "Decade", "YearOfDecade"};
        final Map<String, String> transformMonthsLivedMap =
                pdfConfig.getCustomData().get("transformMonthsLived");
        // For each `/scheduleEicDependents/[n]/` that exists for n from 0 to 2 ...
        for (int dependentNum = 0; dependentNum <= 2; dependentNum++) {
            // ... handle special case: the PDF has four distinct fields: one for each digit of the birth year.
            // Config maps these to pseudo paths
            // ^\s*\/scheduleEicDependents\/\/[n\/]\/~birth(Millennium|Century|Decade|YearOfDecade)\s*$
            // Here we generate entries with those pseudo paths.
            final Optional<Object> factOptional =
                    facts.getOptional(String.format("%s/[%d]/dateOfBirth", EITC_DEPENDENTS, dependentNum));
            if (factOptional.isPresent()) {
                final String yearString = factOptional.get().toString().substring(0, 4);
                final String[] digitStrings = yearString.split("");
                for (int pos = 0; pos < digitStrings.length && pos < pathSuffixes.length; pos++) {
                    facts.put(
                            String.format("%s/[%d]/~birth%s", EITC_DEPENDENTS, dependentNum, pathSuffixes[pos]),
                            digitStrings[pos]);
                }
            }

            // Transform enum value to digit string
            final Optional<Object> duration =
                    facts.getOptional(String.format("%s/[%d]/monthsLivedWithTPInUS", EITC_DEPENDENTS, dependentNum));
            if (duration.isPresent()) {
                final String transformed =
                        transformMonthsLivedMap.get(duration.get().toString());
                if (transformed != null) {
                    facts.put(
                            String.format("%s/[%d]/~transformMonthsLived", EITC_DEPENDENTS, dependentNum), transformed);
                }
            }
        }
        return true;
    }
}
