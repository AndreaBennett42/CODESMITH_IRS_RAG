package gov.irs.directfile.stateapi.repository.facts;

import java.util.ArrayList;
import java.util.HashMap;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import gov.irs.directfile.stateapi.model.GetStateExportedFactsResponse;

@Component
@ConditionalOnProperty(name = "direct-file.exported-facts.mock", havingValue = "true", matchIfMissing = false)
@Slf4j
@SuppressWarnings("PMD.AvoidDuplicateLiterals")
public class ExportedFactsClientMock implements ExportedFactsClient {

    private static final String FILERS = "filers";
    private static final String FAMILY_AND_HH = "familyAndHousehold";
    private static final String FIRST_NAME = "firstName";
    public static final String MIDDLE_INITIAL = "middleInitial";
    public static final String LAST_NAME = "lastName";
    public static final String SUFFIX = "suffix";
    public static final String DATE_OF_BIRTH = "dateOfBirth";
    public static final String RELATIONSHIP = "relationship";
    public static final String ELIGIBLE_DEPENDENT = "eligibleDependent";
    public static final String IS_CLAIMED_DEPENDENT = "isClaimedDependent";
    public static final String IS_PRIMARY_FILER = "isPrimaryFiler";
    private static final String RESIDENCY_DURATION = "residencyDuration";
    private static final String MONTH_LIVE_WITHTP_IN_US = "monthsLivedWithTPInUS";
    private static final String SSN_NOT_VALID_FOR_EMPLOYMENT = "ssnNotValidForEmployment";
    private static final String QUALIFYING_CHILD = "qualifyingChild";
    private static final String EDUCATOR_EXPENSES = "educatorExpenses";
    private static final String HSA_TOTAL_DEDUCTIBLE_AMOUNT = "hsaTotalDeductibleAmount";
    private static final String INTEREST_REPORTS_TOTAL = "interestReportsTotal";
    private static final String FORM_1099Gs_TOTAL = "form1099GsTotal";
    private static final String IS_STUDENT = "isStudent";
    private static final String IS_DISABLED = "isDisabled";

    // 1099-Int
    private static final String TIN_NUM = "tin";
    private static final String RECIPIENT_TIN = "recipientTin";
    private static final String INTEREST_REPORT = "interestReports";
    private static final String AMT_1099 = "1099Amount";
    private static final String TAX_EXEMPT_INTEREST = "taxExemptInterest";
    private static final String INTEREST_GOVERNMENT_BONDS = "interestOnGovernmentBonds";
    private static final String HAS_1099 = "has1099";
    private static final String NO_1099_AMOUNT = "no1099Amount";
    private static final String PAYER_KEY = "payer";
    private static final String PAYER_TIN = "payerTin";
    private static final String TAX_WITHHELD = "taxWithheld";
    private static final String CUSIP_NO = "taxExemptAndTaxCreditBondCusipNo";

    // 1099-G
    private static final String FORM_1099Gs = "form1099Gs";
    private static final String HAS_1099G = "has1099";
    private static final String AMT_1099G = "amount";
    private static final String FED_TAX_WITHHELD = "federalTaxWithheld";
    private static final String STATE_ID_NUM = "stateIdNumber";
    private static final String STATE_TAX_WITHHELD = "stateTaxWithheld";
    private static final String AMT_PAID_FOR_BENEFITS = "amountPaidBackForBenefitsInTaxYear";
    private static final String SCHEDULE_EIC_LINE4A_YES = "scheduleEicLine4aYes";
    private static final String SCHEDULE_EIC_LINE4A_NO = "scheduleEicLine4aNo";
    private static final String SCHEDULE_EIC_LINE4B_YES = "scheduleEicLine4bYes";
    private static final String HOH_QUALIFYING_PERSON = "hohQualifyingPerson";

    // FormW2s
    private static final String FORM_W2s = "formW2s";
    private static final String UNION_DUES_AMT = "unionDuesAmount";
    private static final String BOX14_NJ_UIHCWD = "BOX14_NJ_UIHCWD";
    private static final String BOX14_NJ_UIWFSWF = "BOX14_NJ_UIWFSWF";

    // socialSecurityReports
    private static final String SOCIAL_SECURITY_RPT = "socialSecurityReports";
    private static final String NET_BENEFITS = "netBenefits";
    private static final String FORM_TYPE = "formType";

    @Override
    public Mono<GetStateExportedFactsResponse> getExportedFacts(
            String submissionId, String stateCode, String accountId) {

        log.info("Enter Mock getExportedFacts submissionId={}, accountId={}", submissionId, accountId);
        var exportedFacts = new HashMap<String, Object>();

        var filers = new ArrayList<HashMap<String, Object>>();

        var filer1 = new HashMap<String, Object>();
        filer1.put(FIRST_NAME, "Samuel");
        filer1.put(MIDDLE_INITIAL, null);
        filer1.put(LAST_NAME, "Smith");
        filer1.put(SUFFIX, "Jr");
        filer1.put(DATE_OF_BIRTH, "1985-09-29");
        filer1.put(IS_PRIMARY_FILER, Boolean.TRUE);
        filer1.put(TIN_NUM, "100-01-1234");
        filer1.put(SSN_NOT_VALID_FOR_EMPLOYMENT, false);
        filer1.put(EDUCATOR_EXPENSES, "200.00");
        filer1.put(HSA_TOTAL_DEDUCTIBLE_AMOUNT, "600.00");
        filer1.put(INTEREST_REPORTS_TOTAL, "3000.00");
        filer1.put(FORM_1099Gs_TOTAL, "15000.00");
        filer1.put(IS_STUDENT, false);
        filer1.put(IS_DISABLED, false);
        filers.add(filer1);

        var filer2 = new HashMap<String, Object>();
        filer2.put(FIRST_NAME, "Judy");
        filer2.put(MIDDLE_INITIAL, null);
        filer2.put(LAST_NAME, "Johnson");
        filer2.put(SUFFIX, null);
        filer2.put(DATE_OF_BIRTH, "1985-10-18");
        filer2.put(IS_PRIMARY_FILER, Boolean.FALSE);
        filer2.put(TIN_NUM, "100-02-1234");
        filer2.put(SSN_NOT_VALID_FOR_EMPLOYMENT, false);
        filer2.put(EDUCATOR_EXPENSES, "100.00");
        filer2.put(HSA_TOTAL_DEDUCTIBLE_AMOUNT, "500.00");
        filer2.put(INTEREST_REPORTS_TOTAL, "300.00");
        filer2.put(FORM_1099Gs_TOTAL, "1500.00");
        filer2.put(IS_STUDENT, false);
        filer2.put(IS_DISABLED, false);
        filers.add(filer2);

        exportedFacts.put(FILERS, filers);

        var familyAndHousehold = new ArrayList<HashMap<String, Object>>();

        var person1 = new HashMap<String, Object>();
        person1.put(FIRST_NAME, "Sammy");
        person1.put(MIDDLE_INITIAL, null);
        person1.put(LAST_NAME, "Smith");
        person1.put(SUFFIX, "I");
        person1.put(DATE_OF_BIRTH, "2013-01-21");
        person1.put(RELATIONSHIP, "biologicalChild");
        person1.put(ELIGIBLE_DEPENDENT, true);
        person1.put(IS_CLAIMED_DEPENDENT, true);
        person1.put(TIN_NUM, "200-01-1234");
        person1.put(RESIDENCY_DURATION, "allYear");
        person1.put(MONTH_LIVE_WITHTP_IN_US, "twelve");
        person1.put(SCHEDULE_EIC_LINE4A_YES, true);
        person1.put(SCHEDULE_EIC_LINE4A_NO, false);
        person1.put(SCHEDULE_EIC_LINE4B_YES, false);
        person1.put(HOH_QUALIFYING_PERSON, true);
        person1.put(SSN_NOT_VALID_FOR_EMPLOYMENT, false);
        person1.put(QUALIFYING_CHILD, true);
        familyAndHousehold.add(person1);

        exportedFacts.put(FAMILY_AND_HH, familyAndHousehold);

        var intReports = new ArrayList<HashMap<String, Object>>();
        var intRpt = new HashMap<String, Object>();
        intRpt.put(HAS_1099, Boolean.TRUE);
        intRpt.put(AMT_1099, "800.00");
        intRpt.put(INTEREST_GOVERNMENT_BONDS, "300");
        intRpt.put(TAX_EXEMPT_INTEREST, "200.00");
        intRpt.put(RECIPIENT_TIN, "123-45-6789");
        intRpt.put(NO_1099_AMOUNT, null);
        intRpt.put(PAYER_KEY, "JPM Bank");
        intRpt.put(PAYER_TIN, "01-1234567");
        intRpt.put(TAX_WITHHELD, "120");
        intRpt.put(CUSIP_NO, "01234567A");
        intReports.add(intRpt);

        exportedFacts.put(INTEREST_REPORT, intReports);

        // 1099-G
        var form1099Gs = new ArrayList<HashMap<String, Object>>();
        var form1099G = new HashMap<String, Object>();
        form1099G.put(HAS_1099G, Boolean.TRUE);
        form1099G.put(RECIPIENT_TIN, "123-45-6789");
        form1099G.put(PAYER_KEY, "State of california");
        form1099G.put(PAYER_TIN, "321-54-9876");
        form1099G.put(AMT_1099G, "100.00");
        form1099G.put(FED_TAX_WITHHELD, "20.00");
        form1099G.put(STATE_ID_NUM, "123456");
        form1099G.put(STATE_TAX_WITHHELD, "10.00");
        form1099G.put(AMT_PAID_FOR_BENEFITS, "25.00");
        form1099Gs.add(form1099G);

        exportedFacts.put(FORM_1099Gs, form1099Gs);

        // formW2
        var formW2s = new ArrayList<HashMap<String, Object>>();
        var formW2 = new HashMap<String, Object>();
        formW2.put(UNION_DUES_AMT, "40.00");
        formW2.put(BOX14_NJ_UIHCWD, "101.00");
        formW2.put(BOX14_NJ_UIWFSWF, "101.00");
        formW2s.add(formW2);

        exportedFacts.put(FORM_W2s, formW2s);

        // socialSecurityReports
        var ssRpts = new ArrayList<HashMap<String, Object>>();
        var ssRpt = new HashMap<String, Object>();
        ssRpt.put(RECIPIENT_TIN, "123-45-6789");
        ssRpt.put(NET_BENEFITS, "21000.00");
        ssRpt.put(FORM_TYPE, "SSA-1099");
        ssRpts.add(ssRpt);

        exportedFacts.put(SOCIAL_SECURITY_RPT, ssRpts);

        var response = new GetStateExportedFactsResponse(exportedFacts);

        return Mono.just(response);
    }
}
