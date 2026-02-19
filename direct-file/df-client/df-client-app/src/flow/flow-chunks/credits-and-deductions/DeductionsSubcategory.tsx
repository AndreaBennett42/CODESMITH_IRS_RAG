/* eslint-disable max-len */
import { Assertion, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  FactAssertion,
  Boolean,
  ConditionalList,
  ContextHeading,
  DFModal,
  Dollar,
  Enum,
  Heading,
  IconDisplay,
  InfoDisplay,
  SaveAndOrContinueButton,
  SetFactAction,
  SummaryTable,
  DFAlert,
  KnockoutButton,
} from '../../ContentDeclarations.js';

export const DeductionsSubcategory = (
  <Subcategory
    route='deductions'
    completeIf='/deductionsSectionComplete'
    dataItems={[
      {
        itemKey: `standardDeduction`,
        conditions: [`/wantsStandardDeduction`],
      },
      {
        itemKey: `adjustment`,
        conditions: [`/hasAdjustments`],
      },
    ]}
  >
    <Assertion
      type='info'
      i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./isMFJDependent'
      condition='/isMFJDependent'
    />
    <Screen route='deductions-intro-mfj-dependents' condition='/isMFJDependent'>
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/deductions' />
      <Heading i18nKey='/heading/credits-and-deductions/deductions-intro-mfj-dependent' />
      <DFModal i18nKey='/info/credits-and-deductions/deductions-intro-shared' />
      <DFModal i18nKey='/info/credits-and-deductions/deductions-intro-mfj-dep' />
      <SaveAndOrContinueButton />
    </Screen>
    <Screen route='deductions-intro' condition={{ operator: `isFalseOrIncomplete`, condition: `/isMFJDependent` }}>
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/deductions' />
      <Heading i18nKey='/heading/credits-and-deductions/deductions-intro-mfj' condition='/isFilingStatusMFJ' />
      <Heading
        i18nKey='/heading/credits-and-deductions/deductions-intro-self'
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
      />
      <DFModal i18nKey='/info/credits-and-deductions/deductions-intro-shared' />
      <DFModal i18nKey='/info/credits-and-deductions/deductions-intro-adjustments' />
      <SaveAndOrContinueButton />
    </Screen>
    <SubSubcategory route='hsa-deduction' editable={false}>
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./hsaDeductionAmountPrimary'
        conditions={[`/isFilingStatusSingle`, `/primaryFiler/hasHsaDeduction`]}
      />
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./hsaDeductionAmountPrimary'
        conditions={[
          `/isFilingStatusMFJ`,
          `/primaryFiler/hasHsaDeduction`,
          { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/hasHsaDeduction` },
        ]}
      />
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./hsaDeductionAmountSpouse'
        conditions={[
          `/isFilingStatusMFJ`,
          `/secondaryFiler/hasHsaDeduction`,
          { operator: `isFalse`, condition: `/primaryFiler/hasHsaDeduction` },
        ]}
      />
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./hsaDeductionAmountBoth'
        conditions={[`/isFilingMfJAndBothFilersHaveHsaDeductions`]}
      />
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./hsaDeductionNotQualified'
        conditions={[
          `/isFilingStatusMFJ`,
          { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/hasHsaDeduction` },
          { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/hasHsaDeduction` },
          { operator: `isFalseOrIncomplete`, condition: `/isFilingMfJAndBothFilersHaveHsaDeductions` },
        ]}
      />
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./hsaDeductionNotQualified'
        conditions={[
          `/isFilingStatusSingle`,
          { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/hasHsaDeduction` },
        ]}
      />

      <Screen route='hsa-deduction-intro-qualified' condition='/hasHsaDeduction'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/hsa-deductions'
          batches={[`hsa-0`]}
        />
        <Heading i18nKey='/heading/credits-and-deductions/hsa-qualified-deductions' batches={[`hsa-0`]} />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/deductions-hsa-qualified-amount'
          conditions={[
            `/primaryFiler/hasHsaDeduction`,
            { operator: `isFalse`, condition: `/isFilingMfJAndBothFilersHaveHsaDeductions` },
          ]}
          batches={[`hsa-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/deductions-hsa-qualified-amount-spouse-only'
          conditions={[
            `/secondaryFiler/hasHsaDeduction`,
            { operator: `isFalse`, condition: `/isFilingMfJAndBothFilersHaveHsaDeductions` },
          ]}
          batches={[`hsa-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/deductions-hsa-qualified-amount-mfj-both'
          conditions={[`/isFilingMfJAndBothFilersHaveHsaDeductions`]}
          batches={[`hsa-0`]}
        />
        <SummaryTable
          i18nKey='/info/credits-and-deductions/deductions-hsa-qualified-summary'
          batches={[`hsa-0`]}
          items={[
            {
              itemKey: `hsaDeductions`,
              conditions: [`/primaryFiler/hasHsaDeduction`],
            },
            {
              itemKey: `hsaDeductionsSpouse`,
              conditions: [`/secondaryFiler/hasHsaDeduction`],
            },
            {
              itemKey: `hsaDeductionsTotal`,
            },
          ]}
        />
        <InfoDisplay i18nKey='/info/credits-and-deductions/deductions-hsa-w2-already-excluded' batches={[`hsa-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen
        route='hsa-deduction-intro-not-qualified'
        condition={{ operator: `isFalse`, condition: `/hasHsaDeduction` }}
      >
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/hsa-deductions'
          batches={[`hsa-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/hsa-no-qualified-deductions'
          batches={[`hsa-0`]}
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/hsa-no-qualified-deductions-mfj'
          batches={[`hsa-0`]}
          condition={`/isFilingStatusMFJ`}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/deductions-hsa-no-qualified-amount'
          batches={[`hsa-0`]}
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/deductions-hsa-no-qualified-amount-mfj'
          batches={[`hsa-0`]}
          condition={`/isFilingStatusMFJ`}
        />
        <InfoDisplay i18nKey='/info/credits-and-deductions/deductions-hsa-w2-already-excluded' batches={[`hsa-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/isMFJDependent` }}>
      <SubSubcategory route='educator-expenses-deduction'>
        <Assertion
          type='info'
          i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./educatorExpensesAdjustment'
          conditions={[
            `/flowShowEducatorExpensesSummary`,
            { operator: `isComplete`, condition: `/educatorExpensesAdjustment` },
          ]}
        />
        <Screen route='eligible-educator'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/eligible-educator-context' />
          <Heading i18nKey='/heading/credits-and-deductions/eligible-educator-mfj' condition='/isFilingStatusMFJ' />
          <Heading
            i18nKey='/heading/credits-and-deductions/eligible-educator'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <DFModal
            i18nKey='/info/credits-and-deductions/eligible-educator-details-mfj'
            condition='/isFilingStatusMFJ'
          />
          <DFModal
            i18nKey='/info/credits-and-deductions/eligible-educator-details-self'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Enum path='/wasK12Educators' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='secondary-educator-expenses' condition='/spouseWasK12Educator'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/eligible-educator-context' />
          <Heading i18nKey='/heading/credits-and-deductions/secondary-educator-expenses' />
          <DFModal i18nKey='/info/credits-and-deductions/educator-expense-details' />
          <Dollar path='/secondaryEducatorExpensesWritable' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='primary-educator-expenses' condition='/tpWasK12Educator'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/eligible-educator-context' />
          <Heading i18nKey='/heading/credits-and-deductions/primary-educator-expenses' />
          <DFModal i18nKey='/info/credits-and-deductions/educator-expense-details' />
          <Dollar path='/primaryEducatorExpensesWritable' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Screen route='educator-expenses-summary' condition='/flowShowEducatorExpensesSummary'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/eligible-educator-context' />
        <Heading i18nKey='/heading/credits-and-deductions/eligible-educator-summary' />
        <ConditionalList
          condition={`/isFilingStatusMFJ`}
          i18nKey='/info/credits-and-deductions/educator-expenses-summary-box-primary-and-secondary'
          items={[{ itemKey: `primary` }, { itemKey: `secondary` }]}
          isSummary
        />
        <InfoDisplay i18nKey='/info/credits-and-deductions/educator-expenses-summary' />
        <SaveAndOrContinueButton />
      </Screen>
      <Gate condition='/flowShowStudentLoanInterestSection'>
        <SubSubcategory route='student-loan-int-deduction'>
          <Assertion
            type='info'
            i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./studentLoanInterestAmount'
            conditions={[
              `/flowShowStudentLoansAmount`,
              { operator: `isComplete`, condition: `/studentLoanInterestAmount` },
            ]}
          />
          <Screen route='paid-student-loan-interest'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/student-loan-interest-deduction'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/paid-student-loan-interest'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/paid-student-loan-interest-mfj'
              condition='/isFilingStatusMFJ'
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/paid-student-loan-interest-details'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/paid-student-loan-interest-details-mfj'
              condition='/isFilingStatusMFJ'
            />
            <Boolean path='/hadStudentLoanInterestPayments' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
            <Boolean
              path='/hadStudentLoanInterestPayments'
              i18nKeySuffixContext='self'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='student-loans-qualify' condition='/hadStudentLoanInterestPayments'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/student-loan-interest-deduction'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/student-loans-qualify'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/student-loans-qualify-mfj'
              condition='/isFilingStatusMFJ'
            />
            <ConditionalList
              i18nKey='/info/credits-and-deductions/student-loans-qualify-details'
              items={[
                { itemKey: `listItem_whom`, conditions: [`/isFilingStatusSingle`] },
                {
                  itemKey: `listItem_whom_non_single_mfj`,
                  conditions: [{ operator: `isFalse`, condition: `/isFilingStatusSingle` }, `/isFilingStatusMFJ`],
                },
                {
                  itemKey: `listItem_whom_non_single`,
                  conditions: [
                    { operator: `isFalse`, condition: `/isFilingStatusSingle` },
                    { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  ],
                },
                { itemKey: `listItem_program` },
                { itemKey: `listItem_time` },
              ]}
            />
            <Boolean path='/studentLoansQualify' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='student-loan-interest-amount' condition='/flowShowStudentLoansAmount'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/student-loan-interest-deduction'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/student-loan-amount'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading i18nKey='/heading/credits-and-deductions/student-loan-amount-mfj' condition='/isFilingStatusMFJ' />
            <InfoDisplay i18nKey='/info/credits-and-deductions/student-loan-amount-details' />
            <DFModal i18nKey='/info/credits-and-deductions/student-loan-amount-snack' />
            <Dollar path='/studentLoanInterestAmount' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='student-loan-interest-summary' condition='/flowShowStudentLoansAmount'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/student-loan-interest-deduction'
            />
            <Heading i18nKey='/heading/credits-and-deductions/student-loan-interest-summary' />
            <ConditionalList
              i18nKey='/info/credits-and-deductions/student-loan-interest-summary'
              isSummary
              items={[{ itemKey: `paid` }, { itemKey: `deduct` }]}
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/student-loan-interest-summary-2' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Gate>
      <Screen route='student-loan-interest-not-qualified' condition='/cannotDeductStudentLoanInterest'>
        {/* Should only show one rejection reason */}
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/student-loan-interest-deduction'
        />
        {/* Not qualified because of MFS filing status */}
        <Heading
          i18nKey='/heading/credits-and-deductions/student-loan-interest-not-qualified'
          condition='/isFilingStatusMFS'
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/student-loans-not-qualified-mfs'
          condition='/isFilingStatusMFS'
        />
        {/* Not qualified because of income  */}
        <Heading
          i18nKey='/heading/credits-and-deductions/student-loan-interest-not-qualified'
          conditions={[
            `/cannotDeductStudentLoanInterestBecauseIncomeOrFilingStatus`,
            { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
            { operator: `isFalse`, condition: `/isFilingStatusMFS` },
          ]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/student-loan-interest-not-qualified-mfj'
          conditions={[`/cannotDeductStudentLoanInterestBecauseIncomeOrFilingStatus`, `/isFilingStatusMFJ`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/student-loans-not-qualified-income'
          conditions={[
            `/incomeTooHighForStudentLoanInterestDeductionNotMFJ`,
            { operator: `isFalse`, condition: `/isFilingStatusMFS` },
          ]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/student-loans-not-qualified-income-mfj'
          condition='/incomeTooHighForStudentLoanInterestDeductionMFJ'
        />
        {/* Not qualified because someone is claiming as a dependent */}
        <Heading
          i18nKey='/heading/credits-and-deductions/student-loans-not-qualified-dependents'
          condition='/cannotDeductStudentLoanInterestBecauseDependents'
        />
        {/* Not qualified because loans don't qualify */}
        <Heading
          i18nKey='/heading/credits-and-deductions/student-loans-not-qualified-loans-not-qualified'
          condition='/cannotDeductStudentLoanInterestBecauseLoansDontQualify'
        />
        <SaveAndOrContinueButton />
      </Screen>
    </Gate>

    <SubSubcategory route='standard-deduction' editable={false}>
      <Screen route='standard-deduction-intro'>
        <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/standard-deduction-intro-context'
        />
        <Heading i18nKey='/heading/credits-and-deductions/standard-deduction-intro' />
        <InfoDisplay i18nKey='/info/credits-and-deductions/standard-deduction-intro-details' />
        <ConditionalList
          i18nKey='/info/credits-and-deductions/standard-deduction-intro-details'
          items={[
            { itemKey: `filing_status` },
            {
              itemKey: `tp_age`,
              conditions: [
                `/primaryFiler/age65OrOlder`,
                { operator: `isFalseOrIncomplete`, condition: `/secondaryFilerIs65OrOlderForDeduction` },
              ],
            },
            {
              itemKey: `sp_age`,
              conditions: [
                `/isSpouseInfoApplicableForStandardDeductions`,
                `/secondaryFilerIs65OrOlderForDeduction`,
                { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/age65OrOlder` },
              ],
            },
            {
              itemKey: `tp_sp_age`,
              conditions: [
                `/isSpouseInfoApplicableForStandardDeductions`,
                `/secondaryFilerIs65OrOlderForDeduction`,
                `/primaryFiler/age65OrOlder`,
              ],
            },
            {
              itemKey: `tp_blind`,
              conditions: [
                `/primaryFiler/isBlind`,
                { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/isBlind` },
              ],
            },
            {
              itemKey: `sp_blind`,
              conditions: [
                `/isSpouseInfoApplicableForStandardDeductions`,
                `/secondaryFiler/isBlind`,
                { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/isBlind` },
              ],
            },
            {
              itemKey: `tp_sp_blind`,
              conditions: [
                `/isSpouseInfoApplicableForStandardDeductions`,
                `/secondaryFiler/isBlind`,
                `/primaryFiler/isBlind`,
              ],
            },
            {
              itemKey: `tp_can_be_claimed`,
              conditions: [
                `/primaryFiler/canBeClaimed`,
                { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/canBeClaimed` },
              ],
            },
            {
              itemKey: `sp_can_be_claimed`,
              conditions: [
                `/isSpouseInfoApplicableForStandardDeductions`,
                `/secondaryFiler/canBeClaimed`,
                { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/canBeClaimed` },
              ],
            },
            {
              itemKey: `tp_sp_can_be_claimed`,
              conditions: [
                `/isSpouseInfoApplicableForStandardDeductions`,
                `/secondaryFiler/canBeClaimed`,
                `/primaryFiler/canBeClaimed`,
              ],
            },
          ]}
        />
        <FactAssertion
          type='info'
          displayOnlyOn='data-view'
          i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./standardDeduction'
          conditions={[
            { operator: `isComplete`, condition: `/hasHsaDeduction` },
            { operator: `isComplete`, condition: `/educatorExpensesAdjustment` },
            { operator: `isComplete`, condition: `/studentLoanInterestAdjustmentAmount` },
          ]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <SubSubcategory route='taxable-income' editable={false}>
      <Screen route='taxable-income-summary'>
        <Heading i18nKey='/heading/credits-and-deductions/taxable-income-summary' />
        <SummaryTable
          i18nKey='/info/credits-and-deductions/taxable-income-summary-total-income'
          items={[{ itemKey: `totalIncome` }]}
        />
        <SummaryTable
          i18nKey='/info/credits-and-deductions/taxable-income-summary'
          items={[
            {
              itemKey: `adjustmentsHeader`,
            },
            {
              itemKey: `hsaDeduction`,
              indent: true,
            },
            {
              itemKey: `educatorExpenses`,
              indent: true,
            },
            {
              itemKey: `studentLoans`,
              indent: true,
            },
            {
              itemKey: `agi`,
              showTopBorder: true,
            },
            {
              itemKey: `standardDeduction`,
            },
            {
              itemKey: `taxableIncome`,
              showTopBorder: true,
            },
          ]}
        />
        <InfoDisplay i18nKey='/info/credits-and-deductions/taxable-income-summary' />
        <FactAssertion
          type='info'
          displayOnlyOn='data-view'
          i18nKey='dataviews./flow/credits-and-deductions/deductions.assertions./taxableIncome'
          conditions={[
            { operator: `isComplete`, condition: `/hasHsaDeduction` },
            { operator: `isComplete`, condition: `/educatorExpensesAdjustment` },
            { operator: `isComplete`, condition: `/studentLoanInterestAdjustmentAmount` },
          ]}
        />
        <SetFactAction path='/flowHasSeenDeductions' source='/flowTrue' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='net-investment-mfs-ko' condition='/hasInterestIncomeAndAgiAboveThresholdForMfs' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/forms-missing/net-investment' />
        <InfoDisplay i18nKey='/info/knockout/net-investment-mfs' />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      <Screen
        route='net-investment-mfj-qss-ko'
        condition='/hasInterestIncomeAndAgiAboveThresholdForMfjOrQss'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/forms-missing/net-investment' />
        <InfoDisplay i18nKey='/info/knockout/net-investment-mfj-qss' />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      <Screen
        route='net-investment-single-hoh-ko'
        condition='/hasInterestIncomeAndAgiAboveThresholdForSingleOrHoh'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/forms-missing/net-investment' />
        <InfoDisplay i18nKey='/info/knockout/net-investment-single-hoh' />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>
  </Subcategory>
);
