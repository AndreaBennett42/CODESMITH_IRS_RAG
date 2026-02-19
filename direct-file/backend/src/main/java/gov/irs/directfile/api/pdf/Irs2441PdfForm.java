package gov.irs.directfile.api.pdf;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.Vector;

import lombok.extern.slf4j.Slf4j;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

@Slf4j
public class Irs2441PdfForm extends PdfForm {
    public Irs2441PdfForm(final String templateName) {
        super(templateName);
    }

    static String PATH_FORMAT = "%s/#%s/%s";
    static String CLAIMING_CREDIT = "/cdccQualified";

    private void format2441Strings(final FactEvaluationResult facts) {
        // Pending an 'AsInt' CompNode, will update the formatting of this fact manually
        // Pulling the XML fact since it's already been casted to a string in the FG
        final String CARE_EXPENSES_DECIMAL_AMOUNT = "/xmlCdccCareExpensesDecimalAmount";
        final String PSEUDO_CARE_EXPENSES_DECIMAL_PATH = "/~pdfCdccCareExpensesDecimalAmount";
        final Optional<Object> careExpensesDecimalAmount = facts.getOptional(CARE_EXPENSES_DECIMAL_AMOUNT);
        final Optional<Object> isClaimingCdcc = facts.getOptional(CLAIMING_CREDIT);
        final Boolean isClaimingCdccCredit = isClaimingCdcc.isPresent() && (Boolean) isClaimingCdcc.get();
        if (careExpensesDecimalAmount.isPresent() && isClaimingCdccCredit) {
            final String careExpensesDecimalString = (String) careExpensesDecimalAmount.get();
            final String careExpenseDecimalStringFormatted = careExpensesDecimalString.substring(2);
            facts.put(PSEUDO_CARE_EXPENSES_DECIMAL_PATH, careExpenseDecimalStringFormatted);
        }
    }

    private void computeAllQualifyingPeople(final FactEvaluationResult facts) {
        final String[] CDCC_QUALIFYING_PERSON_FACTS = {
            "firstName", "lastName", "tin", "unableToCareForSelfAndOverAge12", "cdccQualifyingExpenseAmount"
        };

        final String[] CDCC_CLAIMING_FACTS = {
            "cdccHasMorethanThreeQualifyingPersons",
            "cdccQualifyingExpenses",
            "cdccEarnedIncomePrimaryFilerLine4",
            "cdccEarnedIncomeLine5",
            "cdccLowerOfEarnedIncomeAndExpenses",
            "cdccCareExpensesDecimalAmount",
            "cdccTentativeExpenseAmount",
            "cdccCreditForPriorYearExpenses",
            "cdccTotalEligibleCDCCAmount",
            "cdccCreditLimit",
            "cdccTotalCredit",
            "form2441Line27ExpenseCap",
            "cdccSumOfDeductibleAndExcludedBenefitsAmount",
            "cdccNetAllowableAmount",
            "form2441Line30Expenses",
            "cdccSmallerOfTotalQualifyingExpensesAmount",
            "agi"
        };

        final Optional<Object> isClaimingCdcc = facts.getOptional(CLAIMING_CREDIT);
        final Boolean isClaimingCdccCredit = isClaimingCdcc.isPresent() && (Boolean) isClaimingCdcc.get();

        if (isClaimingCdccCredit) {
            for (final String factPath : CDCC_CLAIMING_FACTS) {
                var pseudoFactPath = String.format("/~%s", factPath);
                String factSource = String.format("/%s", factPath);
                if ("agi".equals(factPath)) {
                    pseudoFactPath = "/~cdccAgi";
                }
                var sourceFactValue = facts.getString(factSource);
                facts.put(pseudoFactPath, sourceFactValue);
            }
        }

        final String PDF_QUALIFYING_PEOPLE = "/~cdccAllQualifyingPeople";
        final String CDCC_QUALIFYING_FILERS = "/cdccQualifyingFilers";
        final String CDCC_QUALIFYING_DEPENDENTS = "/cdccQualifyingPeople";
        final Optional<Object> qualifyingFilers = facts.getOptional(CDCC_QUALIFYING_FILERS);
        final Optional<Object> qualifyingDependents = facts.getOptional(CDCC_QUALIFYING_DEPENDENTS);
        List<UUID> qualifyingPersonUUIDs = new Vector<UUID>();

        if (qualifyingFilers.isPresent() && isClaimingCdccCredit) {
            final List<UUID> qualifyingFilerIds = (List<UUID>) qualifyingFilers.get();
            for (final UUID qualifyingFilerId : qualifyingFilerIds) {
                qualifyingPersonUUIDs.add(qualifyingFilerId);
                for (final String factPath : CDCC_QUALIFYING_PERSON_FACTS) {
                    var pseudoFactPath = String.format(PATH_FORMAT, PDF_QUALIFYING_PEOPLE, qualifyingFilerId, factPath);

                    String factSource = String.format(PATH_FORMAT, CDCC_QUALIFYING_FILERS, qualifyingFilerId, factPath);
                    if ("unableToCareForSelfAndOverAge12".equals(factPath)) {
                        // Filer collection doesn't have this fact, use "isDisabled" instead
                        factSource =
                                String.format(PATH_FORMAT, CDCC_QUALIFYING_FILERS, qualifyingFilerId, "isDisabled");
                        var booleanFactValue = facts.getBoolean(factSource);
                        facts.put(pseudoFactPath, booleanFactValue);
                        continue;
                    }
                    var sourceFactValue = facts.getString(factSource);
                    facts.put(pseudoFactPath, sourceFactValue);
                }
            }
        }

        if (qualifyingDependents.isPresent() && isClaimingCdccCredit) {
            final List<UUID> qualifyingDependentIds = (List<UUID>) qualifyingDependents.get();
            for (final UUID qualifyingDependentId : qualifyingDependentIds) {
                qualifyingPersonUUIDs.add(qualifyingDependentId);
                for (final String factPath : CDCC_QUALIFYING_PERSON_FACTS) {
                    var pseudoFactPath =
                            String.format(PATH_FORMAT, PDF_QUALIFYING_PEOPLE, qualifyingDependentId, factPath);

                    String factSource =
                            String.format(PATH_FORMAT, CDCC_QUALIFYING_DEPENDENTS, qualifyingDependentId, factPath);
                    if ("unableToCareForSelfAndOverAge12".equals(factPath)) {
                        var booleanFactValue = facts.getBoolean(factSource);
                        facts.put(pseudoFactPath, booleanFactValue);
                        continue;
                    }
                    var sourceFactValue = facts.getString(factSource);
                    facts.put(pseudoFactPath, sourceFactValue);
                }
            }
        }

        // Add UUIDs to pseudo collection
        facts.put(PDF_QUALIFYING_PEOPLE, qualifyingPersonUUIDs);
    }

    private void computeCareProviders(final FactEvaluationResult facts) {
        // reassigns pdf-related care provider facts to a pseudo care provider collection. Because we cannot
        // conditionally
        // apply a fact in the configuration.yml, we use conditionally existent pseudo care provider fact values in the
        // yml
        // instead of the actual values.
        final String[] CARE_PROVIDERS_FACTS = {
            "displayName",
            "pdfAddress",
            "pdfTinEinColumn",
            "isHouseholdEmployee",
            "isHouseholdEmployeeNo",
            "amountPaidForCare"
        };
        final String ACTUAL_CARE_PROVIDERS_PATH = "/cdccCareProviders";
        final String PSEUDO_CARE_PROVIDERS_PATH = "/~careProviders";
        final Optional<Object> careProviders = facts.getOptional(ACTUAL_CARE_PROVIDERS_PATH);

        if (careProviders.isPresent()) {
            final List<UUID> careProviderIds = (List<UUID>) careProviders.get();
            for (final UUID careProviderId : careProviderIds) {
                for (final String factPath : CARE_PROVIDERS_FACTS) {
                    var actualFactPath =
                            String.format(PATH_FORMAT, ACTUAL_CARE_PROVIDERS_PATH, careProviderId, factPath);
                    var pseudoFactPath =
                            String.format(PATH_FORMAT, PSEUDO_CARE_PROVIDERS_PATH, careProviderId, factPath);
                    if ("isHouseholdEmployee".equals(factPath) || "isHouseholdEmployeeNo".equals(factPath)) {
                        facts.put(pseudoFactPath, facts.getBoolean(actualFactPath));
                        log.info("ACTUAL FACT VALUE: ", facts.getBoolean(actualFactPath));
                    } else {
                        facts.put(pseudoFactPath, facts.getString(actualFactPath));
                    }
                }
            }
            // create pseudo care provider collection
            facts.put(PSEUDO_CARE_PROVIDERS_PATH, careProviderIds);
        }
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        final String CDCC_CARE_PROVIDERS = "/cdccCareProviders";

        if (!facts.getBoolean("/shouldIncludeCareProviders")) {
            // line 1(a), row 1
            var noneString = facts.getString("/pdfNone");
            facts.put("~form2441CareProviderNameOrNone", noneString);
        } else {
            // line 1(a), row 1
            final String providerName = facts.getString(String.format("%s/[0]/displayName", CDCC_CARE_PROVIDERS));
            facts.put("~form2441CareProviderNameOrNone", providerName);
            computeCareProviders(facts);
        }

        // Compute All CDCC Qualifying People (Filers + Dependents and Nondependents)
        computeAllQualifyingPeople(facts);
        // Perform any necessary manual string formatting
        format2441Strings(facts);
        return true;
    }
}
