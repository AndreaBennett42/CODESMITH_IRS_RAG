import { Subcategory, Screen, SubSubcategory, Gate, Assertion } from '../../flowDeclarations.js';
import {
  Heading,
  InfoDisplay,
  DFModal,
  SaveAndOrContinueButton,
  Dollar,
  Boolean,
  InternalLink,
  DFAlert,
  IconDisplay,
  KnockoutButton,
  TaxReturnAlert,
  Enum,
  SaveAndOrContinueAndSetFactButton,
  ConditionalList,
} from '../../ContentDeclarations.js';
import {
  cdccSubSectionNames,
  earnedIncomeRuleKnockoutCheckFlow,
} from '../credits-and-deductions/SpecialIncomeRuleKnockoutFlow.js';
import { BATCH_NAME } from '../../batches.js';
import { makeCdccCareProvidersCollection } from '../credits-and-deductions/CdccCareProvidersCollection.js';

export const makeHouseholdEmployeeCheckScreen = (suffix: string, batches: BATCH_NAME[]) => (
  <Screen route={`household-employee-check-${suffix}`}>
    <Heading
      condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
      i18nKey='/heading/income/dependent-care/household-employee-check'
      batches={batches}
    />
    <Heading
      condition='/isFilingStatusMFJ'
      i18nKey='/heading/income/dependent-care/household-employee-check-mfj'
      batches={batches}
    />
    <DFModal i18nKey='/info/income/dependent-care/household-employee-check' />
    <InfoDisplay i18nKey='/info/income/dependent-care/learn-more-about-household-employees' batches={batches} />
    <Boolean path='/flowKnockoutHouseholdEmployee' batches={[`cdcc-2`]} />

    <SaveAndOrContinueButton />
  </Screen>
);

export const makeHouseholdEmployeeKnockoutScreen = (suffix: string, batches: BATCH_NAME[]) => (
  <Screen route={`household-employee-ko-${suffix}`} condition='/flowKnockoutHouseholdEmployee' isKnockout={true}>
    <IconDisplay name='ErrorOutline' size={9} isCentered />
    <Heading i18nKey='/heading/income/dependent-care/household-employee-ko' batches={batches} />
    <InfoDisplay i18nKey='/info/income/dependent-care/household-employee-ko' batches={batches} />
    <InfoDisplay i18nKey='/info/income/dependent-care/learn-more-about-household-employees' batches={batches} />
    <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />

    <KnockoutButton i18nKey='button.knockout' />
  </Screen>
);

export const DependentCareBenefitsSubcategory = (
  <Subcategory
    route='dependent-care'
    completeIf='/dependentCareBenefitsIsComplete'
    dataItems={[
      {
        itemKey: `dependentCareTaxable`,
        conditions: [`/hasReportedDependentCareBenefits`],
      },
      {
        itemKey: `dependentCareNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/hasReportedDependentCareBenefits` }],
      },
    ]}
  >
    <Screen route='dep-care-intro'>
      <Heading i18nKey='/heading/income/dependent-care/dependent-care-intro' batches={[`dependent-care-benefits-0`]} />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/dependent-care-intro/no-value-in-w2-box-10-not-mfj'
        conditions={[
          { operator: `isFalse`, condition: `/cdccHasDepCareBenefitsBox10` },
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
        ]}
        batches={[`cdcc-2`]}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/dependent-care-intro/no-value-in-w2-box-10-mfj'
        conditions={[{ operator: `isFalse`, condition: `/cdccHasDepCareBenefitsBox10` }, `/isFilingStatusMFJ`]}
        batches={[`cdcc-2`]}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/dependent-care-intro/has-value-in-w2-box-10-both-filers'
        conditions={[
          `/hasReportedPrimaryFilerDependentCareBenefits`,
          `/hasReportedSecondaryFilerDependentCareBenefits`,
        ]}
        batches={[`cdcc-2`]}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/dependent-care-intro/has-value-in-w2-box-10-primary-only'
        conditions={[
          `/hasReportedPrimaryFilerDependentCareBenefits`,
          { operator: `isFalse`, condition: `/hasReportedSecondaryFilerDependentCareBenefits` },
        ]}
        batches={[`cdcc-2`]}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/dependent-care-intro/has-value-in-w2-box-10-secondary-only'
        conditions={[
          `/hasReportedSecondaryFilerDependentCareBenefits`,
          { operator: `isFalse`, condition: `/hasReportedPrimaryFilerDependentCareBenefits` },
        ]}
        batches={[`cdcc-2`]}
      />
      <DFModal
        i18nKey='/info/income/dependent-care/dependent-care-intro/what-are-dependent-care-benefits'
        batches={[`dependent-care-benefits-0`]}
      />
      <SaveAndOrContinueButton />
    </Screen>
    <SubSubcategory route='dep-care-benefits'>
      {makeHouseholdEmployeeCheckScreen(cdccSubSectionNames.DEPENDENT_CARE_BENEFITS, [
        `dependent-care-benefits-0`,
        `cdcc-0`,
      ])}
      {makeHouseholdEmployeeKnockoutScreen(cdccSubSectionNames.DEPENDENT_CARE_BENEFITS, [
        `dependent-care-benefits-0`,
        `cdcc-0`,
      ])}
      <Screen route='add-dep-care-carry-forward-prior-year-a'>
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-prior-year-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-prior-year-single'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/dep-care-carry-forward-prior-year-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/dep-care-carry-forward-prior-year-single'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <Boolean path='/hasCdccCarryoverAmountFromPriorTaxYear' batches={[`dependent-care-benefits-0`, `cdcc-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='add-dep-care-carry-forward-prior-year-b' condition='/hasCdccCarryoverAmountFromPriorTaxYear'>
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-prior-year-b-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-prior-year-b-single'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <Dollar path='/writableCdccCarryoverAmountFromPriorTaxYear' batches={[`dependent-care-benefits-0`, `cdcc-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='add-dep-care-carry-forward-next-year-a' condition='/cdccHasDepCareBenefitsBox10'>
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-next-year-a-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-next-year-a-single'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/dep-care-carry-forward-next-year-a'
          batches={[`dependent-care-benefits-0`]}
        />
        <Boolean path='/writableHasCdccForfeitedCredits' batches={[`dependent-care-benefits-0`, `cdcc-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='add-dep-care-carry-forward-next-year-b' condition='/hasCdccForfeitedCredits'>
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-next-year-b-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-carry-forward-next-year-b-single'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/dep-care-carry-forward-next-year-a'
          batches={[`dependent-care-benefits-0`]}
        />
        <Dollar path='/writableCdccForfeitedCredits' batches={[`dependent-care-benefits-0`, `cdcc-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='dep-care-mfj-dep-taxpayers' condition='/isMFJDependent'>
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/dep-care-mfj-dep-taxpayers'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/dep-care-mfj-dep-taxpayers-p1'
          batches={[`cdcc-2`]}
        />
        <InternalLink
          i18nKey='/info/credits-and-deductions/credits/dep-care-mfj-dep-taxpayers'
          route='/flow/you-and-your-family/spouse/spouse-mfj-refund-only'
          batches={[`cdcc-2`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <SubSubcategory route='dep-care-expenses'>
      <Gate condition='/cdccShouldBeAskedAboutMaxExclusion'>
        <Screen route='add-dep-care-max-exclusion-tp1' condition='/cdccShouldAskAboutPrimaryFilerEmployerExclusionPlan'>
          <TaxReturnAlert
            type='error'
            i18nKey='/info/income/dependent-care/primary-filer-dependent-plan-exceeds-federal-limit/you'
            condition='/cdccPrimaryFilerDependentCarePlanMaximumExceedsLimit'
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          {/* primary box 10 only */}
          <Heading
            i18nKey='/heading/income/dependent-care/add-dep-care-max-exclusion-tp1/you'
            conditions={[
              `/hasReportedPrimaryFilerDependentCareBenefits`,
              { operator: `isFalse`, condition: `/hasCdccCarryoverAmountFromPriorTaxYear` },
            ]}
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          {/* both primary box 10 and carryover and is single */}
          {/* intentionally uses the same string as above */}
          <Heading
            i18nKey='/heading/income/dependent-care/add-dep-care-max-exclusion-tp1/you'
            conditions={[
              `/hasReportedPrimaryFilerDependentCareBenefits`,
              `/hasCdccCarryoverAmountFromPriorTaxYear`,
              { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
            ]}
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          {/* both primary box 10 and carryover and is mfj */}
          <Heading
            i18nKey='/heading/income/dependent-care/add-dep-care-max-exclusion-tp1/you-and-your-spouse'
            conditions={[
              `/hasReportedPrimaryFilerDependentCareBenefits`,
              `/hasCdccCarryoverAmountFromPriorTaxYear`,
              `/isFilingStatusMFJ`,
            ]}
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          {/* carryover only and is single */}
          <Heading
            i18nKey='/heading/income/dependent-care/add-dep-care-max-exclusion-tp1/you-last-year'
            conditions={[
              { operator: `isFalse`, condition: `/hasReportedPrimaryFilerDependentCareBenefits` },
              { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
            ]}
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          {/* carryover only and is mfj */}
          <Heading
            i18nKey='/heading/income/dependent-care/add-dep-care-max-exclusion-tp1/you-and-your-spouse-last-year'
            conditions={[
              { operator: `isFalse`, condition: `/hasReportedPrimaryFilerDependentCareBenefits` },
              `/isFilingStatusMFJ`,
            ]}
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          <InfoDisplay
            i18nKey='/info/income/dependent-care/add-dep-care-max-exclusion'
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          <Dollar
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            path='/writablePrimaryFilerDependentCarePlanMaximum'
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='add-dep-care-max-exclusion-tp2' condition='/hasReportedSecondaryFilerDependentCareBenefits'>
          <TaxReturnAlert
            type='error'
            i18nKey='/info/income/dependent-care/primary-filer-dependent-plan-exceeds-federal-limit/your-spouse'
            condition='/cdccSecondaryFilerDependentCarePlanMaximumExceedsLimit'
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          <Heading
            i18nKey='/heading/income/dependent-care/add-dep-care-max-exclusion-tp2'
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          <InfoDisplay
            i18nKey='/info/income/dependent-care/add-dep-care-max-exclusion'
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          <Dollar
            path='/writableSecondaryFilerDependentCarePlanMaximum'
            batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='zero-qp-zero-qe' condition={{ operator: `isFalse`, condition: `/cdccHasQualifyingPersons` }}>
          <Heading batches={[`cdcc-1`]} i18nKey='/heading/income/dependent-care/zero-qp-zero-qe' />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/income/dependent-care/zero-qp-zero-qe' />
          <InternalLink
            batches={[`cdcc-1`]}
            i18nKey='/info/income/dependent-care/zero-qp-zero-qe-link'
            route='/data-view/flow/you-and-your-family/dependents'
          />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/income/dependent-care/zero-qp-zero-qe-info' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/cdccHasQualifyingPersons'>
          <Screen route='add-dep-care-expenses' condition='/hasReportedDependentCareBenefits'>
            <Heading
              i18nKey='/heading/income/dependent-care/dependent-care-expenses'
              batches={[`cdcc-2`]}
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/income/dependent-care/dependent-care-expenses-mfj'
              batches={[`cdcc-2`]}
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay
              i18nKey='/info/income/dependent-care/dependent-care-expenses'
              batches={[`cdcc-2`]}
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                `/cdccMoreThanOneQualifyingPerson`,
              ]}
            />
            <InfoDisplay
              i18nKey='/info/income/dependent-care/dependent-care-expenses-mfj'
              batches={[`cdcc-2`]}
              conditions={[`/isFilingStatusMFJ`, `/cdccMoreThanOneQualifyingPerson`]}
            />
            <InfoDisplay
              i18nKey='/info/income/dependent-care/dependent-care-expenses-one-only'
              batches={[`cdcc-2`]}
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/cdccMoreThanOneQualifyingPerson` },
              ]}
            />
            <InfoDisplay
              i18nKey='/info/income/dependent-care/dependent-care-expenses-mfj-one-only'
              batches={[`cdcc-2`]}
              conditions={[
                `/isFilingStatusMFJ`,
                { operator: `isFalse`, condition: `/cdccMoreThanOneQualifyingPerson` },
              ]}
            />
            <ConditionalList
              i18nKey='/info/income/dependent-care/dependent-care-expenses-qps'
              items={[
                { itemKey: `filers`, collection: `/cdccQualifyingFilers` },
                { itemKey: `nonfilers`, collection: `/cdccQualifyingPeople` },
              ]}
              batches={[`cdcc-2`]}
            />
            <InfoDisplay i18nKey='/info/income/dependent-care/dependent-care-expenses-p2' batches={[`cdcc-2`]} />
            <ConditionalList
              i18nKey='/info/income/dependent-care/dependent-care-expenses-exclude'
              items={[
                { itemKey: `turned13`, collection: `/cdccNonFilerQpsWhoTurned13InTaxYearAbleToCareForSelf` },
                { itemKey: `filerQps`, collection: `/cdccQualifyingFilers` },
                { itemKey: `nonFilerQps`, collection: `/cdccNonFilerQpsWhoWereUnableToCareForSelfAndNotQcUnderAge13` },
                {
                  itemKey: `nonFilerQcsTurned13UnableToCareForSelf`,
                  collection: `/cdccNonFilerQcsWhoTurned13InTaxYearUnableToCareForSelf`,
                },
              ]}
              i18nPrefixKey='/info/income/dependent-care/dependent-care-expenses-exclude-prefix'
              condition='/hasQualifyingPersonsWithExpenseRestrictions'
              batches={[`cdcc-2`]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/what-are-dependent-care-expenses'
              batches={[`cdcc-2`]}
            />
            <Dollar path='/writableCdccTotalQualifiedDependentCareExpenses' batches={[`cdcc-2`]} />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route={`dep-care-combat-pay`}>
      <Gate condition='/maybeEligibleForDependentCareBenefits'>
        <Gate condition='/combatPayCouldImpactDependentCareBenefits'>
          <Screen route={`dep-care-combat-pay`}>
            <Heading
              i18nKey='/heading/cdcc-shared/combat-pay/you'
              batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              condition='/onlyPrimaryFilerHasCombatPay'
            />
            <Heading
              i18nKey='/heading/cdcc-shared/combat-pay/your-spouse'
              batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              condition='/onlyMFJSpouseHasCombatPay'
            />
            <Heading
              i18nKey='/heading/cdcc-shared/combat-pay/you-and-your-spouse'
              batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              condition='/bothFilersHasCombatPay'
            />
            <DFAlert i18nKey={null} type='success' headingLevel='h3' batches={[`cdcc-0`, `dependent-care-benefits-0`]}>
              <InfoDisplay
                i18nKey='/info/income/dependent-care/combat-pay-recommendation/you'
                condition='/onlyOneFilerHasCombatPay'
              />
              <InfoDisplay
                i18nKey='/info/income/dependent-care/combat-pay-recommendation/you-and-your-spouse'
                condition='/bothFilersHasCombatPay'
              />
              <DFModal
                i18nKey='/info/income/dependent-care/combat-pay-modal'
                batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              />
            </DFAlert>

            <SaveAndOrContinueAndSetFactButton
              i18nKey='button.continueIncludeCombatPayCdcc'
              sourcePath='/cdccCombatPayRecommendation'
              destinationPath='/cdccCombatPayElectionForBenefits'
              conditions={[`/someFilerHasCombatPay`, { operator: `isFalse`, condition: `/bothFilersHasCombatPay` }]}
            />
            <SaveAndOrContinueAndSetFactButton
              i18nKey='button.continueIncludeBothCombatPayCdcc'
              sourcePath='/cdccCombatPayRecommendation'
              destinationPath='/cdccCombatPayElectionForBenefits'
              condition='/bothFilersHasCombatPay'
            />
            <InternalLink
              i18nKey='/info/cdcc-shared/combat-pay-change/another-option-link'
              route={`/flow/income/dependent-care/dep-care-combat-pay-change`}
            />
          </Screen>

          <Screen route={`dep-care-combat-pay-change`} routeAutomatically={false}>
            <Heading
              i18nKey='/heading/income/dep-care-combat-pay-change/you'
              batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              condition='/onlyPrimaryFilerHasCombatPay'
            />
            <Heading
              i18nKey='/heading/income/dep-care-combat-pay-change/your-spouse'
              batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              condition='/onlyMFJSpouseHasCombatPay'
            />
            <Heading
              i18nKey='/heading/income/dep-care-combat-pay-change/you-or-your-spouse'
              batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              condition='/bothFilersHasCombatPay'
            />
            <DFAlert i18nKey={null} type='success' headingLevel='h3' batches={[`cdcc-0`, `dependent-care-benefits-0`]}>
              <InfoDisplay
                i18nKey='/info/income/dependent-care/combat-pay-recommendation/you'
                condition='/onlyOneFilerHasCombatPay'
              />
              <InfoDisplay
                i18nKey='/info/income/dependent-care/combat-pay-recommendation/you-and-your-spouse'
                condition='/bothFilersHasCombatPay'
              />
              <DFModal
                i18nKey='/info/income/dependent-care/combat-pay-modal'
                batches={[`cdcc-0`, `dependent-care-benefits-0`]}
              />
            </DFAlert>

            <Enum path='/cdccCombatPayElectionForBenefits' batches={[`cdcc-2`]} />

            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
        {earnedIncomeRuleKnockoutCheckFlow(
          cdccSubSectionNames.DEPENDENT_CARE_BENEFITS,
          `/showSpecialEarnedIncomeKnockoutFlowInBenefits`,
          [`dependent-care-benefits-0`, `cdcc-0`]
        )}
        <Gate condition='/mfsNotEligibleForCdcc'>
          <Screen route='mfs-earned-income'>
            <Heading
              i18nKey='/heading/income/dependent-care/mfs-earned-income'
              batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            />
            <InfoDisplay
              i18nKey='/info/income/dependent-care/mfs-earned-income'
              batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            />
            <DFModal
              i18nKey='/info/income/dependent-care/what-is-earned-income-modal'
              batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            />
            <Boolean path='/hasMfjSpouseEarnedIncome' batches={[`cdcc-2`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='mfs-earned-income-enter' condition='/hasMfjSpouseEarnedIncome'>
            <Heading
              i18nKey='/heading/income/dependent-care/mfs-earned-income-enter'
              batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            />
            <InfoDisplay
              i18nKey='/info/income/dependent-care/mfs-earned-income'
              batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            />
            <DFModal
              i18nKey='/info/income/dependent-care/what-is-earned-income-modal'
              batches={[`dependent-care-benefits-0`, `cdcc-0`]}
            />
            <Dollar batches={[`dependent-care-benefits-0`, `cdcc-0`]} path='/writableMfsSpouseEarnedIncome' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='dep-care-exclusion-outcomes' editable={false}>
      <Assertion
        type='info'
        i18nKey='dataviews./flow/income/dependent-care.assertions.filerHadNone'
        conditions={[{ operator: `isFalseAndComplete`, condition: `/hasReportedDependentCareBenefits` }]}
      />
      <Assertion
        type='info'
        i18nKey='dataviews./flow/income/dependent-care.assertions.filerReportedAndTaxable'
        conditions={[`/hasReportedDependentCareBenefits`, `/hasTaxableDependentCareBenefits`]}
      />
      <Assertion
        type='info'
        i18nKey='dataviews./flow/income/dependent-care.assertions.filerReportedAndNonTaxable'
        conditions={[
          `/hasReportedDependentCareBenefits`,
          { operator: `isFalseAndComplete`, condition: `/hasTaxableDependentCareBenefits` },
        ]}
      />

      <Screen route='dep-care-exclusion-outcome'>
        <Heading
          i18nKey='/heading/income/dependent-care/dependent-care-exclusion-outcome-reported'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          conditions={[`/hasReportedDependentCareBenefits`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/dependent-care-exclusion-outcome-reported-mfj'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          conditions={[`/isFilingStatusMFJ`, `/hasReportedDependentCareBenefits`]}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/dependent-care-exclusion-outcome-none-reported'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          condition={{ operator: `isFalse`, condition: `/hasReportedDependentCareBenefits` }}
        />
        <DFModal
          i18nKey='/info/income/dependent-care/how-are-taxable-dependent-care-benefits-calculated'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/income/dependent-care/dependent-care-exclusion-outcome'
          batches={[`dependent-care-benefits-0`, `cdcc-0`]}
          condition={`/hasReportedDependentCareBenefits`}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    {makeCdccCareProvidersCollection(`benefits-care-providers`, false, `/cdccQualifiedForBenefit`)}
  </Subcategory>
);
