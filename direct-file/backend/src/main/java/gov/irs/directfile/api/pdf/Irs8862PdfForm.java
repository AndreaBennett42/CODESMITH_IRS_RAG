package gov.irs.directfile.api.pdf;

import java.util.Optional;
import java.util.UUID;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

class Irs8862PdfForm extends PdfForm {
    public Irs8862PdfForm(final String templateName) {
        super(templateName);
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        final String EITC_DEPENDENTS = "/scheduleEicDependents";

        if (facts.getBoolean("/form8862RequiredAndHasQualifyingChildren")) {
            // For each `/scheduleEicDependents/[n]/` that exists for n from 0 to 2 ...
            for (int dependentNum = 0; dependentNum <= 2; dependentNum++) {
                // ... handle special case: the PDF has two distinct fields for birth and death dates: one for month,
                // one for day. Config maps these to pseudo paths. Here we generate entries with those pseudo paths.
                final Optional<Object> birthOptional = facts.getOptional(
                        String.format("%s/[%d]/birthDateIfBornInTaxYear", EITC_DEPENDENTS, dependentNum));
                if (birthOptional.isPresent()) {
                    final String[] dateValues = birthOptional.get().toString().split("-");
                    if (dateValues.length == 3) {
                        facts.put(String.format("%s/[%d]/~dobMonth", EITC_DEPENDENTS, dependentNum), dateValues[1]);
                        facts.put(String.format("%s/[%d]/~dobDay", EITC_DEPENDENTS, dependentNum), dateValues[2]);
                    }
                }

                final Optional<Object> deathOptional = facts.getOptional(
                        String.format("%s/[%d]/deathDateIfDiedInTaxYear", EITC_DEPENDENTS, dependentNum));
                if (deathOptional.isPresent()) {
                    final String[] dateValues = deathOptional.get().toString().split("-");
                    if (dateValues.length == 3) {
                        facts.put(String.format("%s/[%d]/~deathMonth", EITC_DEPENDENTS, dependentNum), dateValues[1]);
                        facts.put(String.format("%s/[%d]/~deathDay", EITC_DEPENDENTS, dependentNum), dateValues[2]);
                    }
                }
            }
        }
        return true;
    }
}
