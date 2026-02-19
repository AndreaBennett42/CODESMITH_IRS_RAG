/* eslint-disable max-len */
import { Assertion, Gate, Screen, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  DFModal,
  DFAlert,
  Heading,
  IconDisplay,
  SaveAndOrContinueButton,
  SaveAndOrContinueAndSetFactButton,
  KnockoutButton,
  InfoDisplay,
  Enum,
  ConditionalList,
  InternalLink,
  ContextHeading,
} from '../../ContentDeclarations.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';
import { makeCdccCareProvidersCollection } from './CdccCareProvidersCollection.js';
import { earnedIncomeRuleKnockoutCheckFlow, cdccSubSectionNames } from './SpecialIncomeRuleKnockoutFlow.js';
import { CdccNonDependentQpTinPinLoop } from './CdccNonDependentQpTinPinLoop.js';
import { makeCdccQualifyingPersonsLoop } from './CdccQualifyingPersonsLoop.js';

export const CdccDisqualifyingItems: ItemConfig[] = [
  {
    itemKey: `subListCdcc-noTaxLiability`,
    conditions: [
      { operator: `isFalse`, condition: `/hasTaxLiabilityBeforeNrCredits` },
      `/maybeEligibleForCdccWoTaxLiability`,
    ],
  },
  {
    itemKey: `subListCdcc-reachedCreditLimit`,
    conditions: [
      `/hasTaxLiabilityBeforeNrCredits`,
      { operator: `isFalse`, condition: `/maybeEligibleForCdcc` },
      `/maybeEligibleForCdccWoTaxLiability`,
    ],
  },
  {
    itemKey: `subListCdcc-noQualifyingPersons`,
    conditions: [
      { operator: `isTrue`, condition: `/isFilingStatusMFJ` },
      { operator: `isFalseAndComplete`, condition: `/cdccHasQualifyingPersons` },
    ],
  },
  {
    itemKey: `subListCdcc-noQualifyingPersonsNotMFJ`,
    conditions: [
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
      { operator: `isFalseAndComplete`, condition: `/cdccHasQualifyingPersons` },
    ],
  },
  {
    itemKey: `subListCdcc-noEarnedIncome`,
    conditions: [
      // prefer the qualifying persons message over this one
      `/cdccHasQualifyingPersons`,
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
      { operator: `isFalseAndComplete`, condition: `/hasEarnedIncomeForCdcc` },
    ],
  },
  {
    itemKey: `subListCdcc-noEarnedIncomeWhenMFJ`,
    conditions: [
      // prefer the qualifying persons message over this one
      `/cdccHasQualifyingPersons`,
      `/isFilingStatusMFJ`,
      { operator: `isFalseAndComplete`, condition: `/hasEarnedIncomeForCdcc` },
    ],
  },
  {
    itemKey: `subListCdcc-noMetExceptions`,
    conditions: [
      // prefer the qualifying persons message over this one
      `/cdccHasQualifyingPersons`,
      // prefer the earned income message over this one
      `/hasEarnedIncomeForCdcc`,
      `/isFilingStatusMFS`,
      { operator: `isFalseAndComplete`, condition: `/mfsButEligibleForCdcc` },
    ],
  },
  {
    itemKey: `subListCdcc-noQualifyingExpenses`,
    conditions: [
      // prefer the qualifying persons message over this one
      `/cdccHasQualifyingPersons`,
      // prefer the earned income message over this one
      `/hasEarnedIncomeForCdcc`,
      // prefer the mfs message over this one
      { operator: `isFalse`, condition: `/mfsNotEligibleForCdcc` },
      { operator: `isTrue`, condition: `/hasTaxLiabilityBeforeNrCredits` },
      { operator: `isFalseOrIncomplete`, condition: `/cdccHasQualifyingExpensesForBenefit` },
      { operator: `isFalseAndComplete`, condition: `/cdccHasQualifyingExpenses` },
    ],
  },
  {
    itemKey: `subListCdcc-noQualifyingExpensesAfterExclusion`,
    conditions: [
      // prefer the qualifying persons message over this one
      `/cdccHasQualifyingPersons`,
      // prefer the earned income message over this one
      `/hasEarnedIncomeForCdcc`,
      // prefer the mfs message over this one
      { operator: `isFalse`, condition: `/mfsNotEligibleForCdcc` },
      { operator: `isTrue`, condition: `/cdccHasQualifyingExpensesForBenefit` },
      { operator: `isFalseAndComplete`, condition: `/cdccHasQualifyingExpenses` },
      { operator: `isFalse`, condition: `/cdccPotentialCdccIsZeroOrLessDueToExclusionBenefits` },
    ],
  },
  {
    itemKey: `subListCdcc-metTheExclusionCap`,
    conditions: [
      // prefer the qualifying persons message over this one
      `/cdccHasQualifyingPersons`,
      // prefer the mfs message over this one
      { operator: `isFalse`, condition: `/mfsNotEligibleForCdcc` },
      { operator: `isTrueAndComplete`, condition: `/cdccHasQualifyingExpensesForBenefit` },
      { operator: `isTrue`, condition: `/cdccPotentialCdccIsZeroOrLessDueToExclusionBenefits` },
    ],
  },
];

export const CdccSubSubcategory = (
  <Gate condition='/maybeEligibleForCdcc'>
    <SubSubcategory route='cdcc-intro' headingLevel='h3' borderStyle='heavy'>
      <Screen route='cdcc-intro'>
        <ContextHeading i18nKey='/heading/credits-and-deductions/credits/cdcc' />
        <Heading i18nKey='/heading/credits-and-deductions/credits/cdcc-intro' batches={[`cdcc-2`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='cdcc-gating'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/cdcc-gating'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/cdcc-gating-spouse'
          condition='/isFilingStatusMFJ'
          batches={[`cdcc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/cdcc-gating'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`cdcc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/cdcc-gating-spouse'
          condition='/isFilingStatusMFJ'
          batches={[`cdcc-0`]}
        />
        <Boolean path='/writableCdccHasQualifyingExpenses' batches={[`cdcc-3`]} />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <Gate condition='/cdccMaybeHasQualifyingExpenses'>
      <SubSubcategory route='cdcc-worksheet-a' headingLevel='h3'>
        <Screen route='worksheet-a-check-1'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading
            batches={[`cdcc-1`]}
            i18nKey='/heading/credits-and-deductions/credits/cdcc/worksheet-a-check-1-mfj'
            condition='/isFilingStatusMFJ'
          />
          <Heading
            batches={[`cdcc-1`]}
            i18nKey='/heading/credits-and-deductions/credits/cdcc/worksheet-a-check-1-single'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/credits-and-deductions/credits/cdcc/worksheet-a-check-1' />
          <Boolean path='/cdccHasCreditForPriorYearExpenses' batches={[`cdcc-3`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='worksheet-a-check-2' condition='/cdccHasCreditForPriorYearExpenses'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-3`]}
          />
          <Heading
            batches={[`cdcc-1`]}
            i18nKey='/heading/credits-and-deductions/credits/cdcc/worksheet-a-check-2-mfj'
            condition='/isFilingStatusMFJ'
          />
          <Heading
            batches={[`cdcc-1`]}
            i18nKey='/heading/credits-and-deductions/credits/cdcc/worksheet-a-check-2-single'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/credits-and-deductions/credits/cdcc/worksheet-a-check-2' />
          <Boolean path='/cdccClaimedCreditForPriorYearExpenses' batches={[`cdcc-2`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='worksheet-a-check-3' condition='/cdccHasPriorYearCreditAndClaimedCredit'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/cdcc/worksheet-a-check-3' />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/credits-and-deductions/credits/cdcc/worksheet-a-check-3' />
          <Boolean path='/cdccClaimedMaxCreditForPriorTaxYear' batches={[`cdcc-2`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='worksheet-a-not-required' condition='/cdccWorksheetANotRequired'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading
            batches={[`cdcc-1`]}
            i18nKey='/heading/credits-and-deductions/credits/cdcc/worksheet-a-not-required'
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='worksheet-a-ko' condition='/knockoutCdccWorksheetA' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/cdcc-generic-ko' batches={[`cdcc-2`]} />
          <InfoDisplay
            batches={[`cdcc-1`]}
            i18nKey='/info/knockout/cdcc-worksheet-a-required-mfj-first'
            condition='/isFilingStatusMFJ'
          />
          <InfoDisplay
            batches={[`cdcc-1`]}
            i18nKey='/info/knockout/cdcc-worksheet-a-required-single-first'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/knockout/cdcc-worksheet-a-required-also-said' />
          <InfoDisplay
            batches={[`cdcc-2`]}
            i18nKey='/info/knockout/cdcc-worksheet-a-required-didnt-claim-credit'
            condition={{ operator: `isFalse`, condition: `/cdccClaimedCreditForPriorYearExpenses` }}
          />
          <InfoDisplay
            batches={[`cdcc-2`]}
            i18nKey='/info/knockout/cdcc-worksheet-a-required-didnt-claim-max'
            condition={{ operator: `isFalse`, condition: `/cdccClaimedMaxCreditForPriorTaxYear` }}
          />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/knockout/cdcc-worksheet-a-required-conclusion' />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/knockout/cdcc-worksheet-a-learn-more' />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
    </Gate>
    <Gate condition='/combatPayCouldImpactCdccCredits'>
      <SubSubcategory route={`cdcc-combat-pay`} headingLevel='h3'>
        <Screen route={`cdcc-combat-pay`}>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading
            i18nKey='/heading/cdcc-shared/combat-pay/you'
            batches={[`cdcc-0`, `cdcc-credits-0`]}
            condition='/onlyPrimaryFilerHasCombatPay'
          />
          <Heading
            i18nKey='/heading/cdcc-shared/combat-pay/your-spouse'
            batches={[`cdcc-0`, `cdcc-credits-0`]}
            condition='/onlyMFJSpouseHasCombatPay'
          />
          <Heading
            i18nKey='/heading/cdcc-shared/combat-pay/you-and-your-spouse'
            batches={[`cdcc-0`, `cdcc-credits-0`]}
            condition='/bothFilersHasCombatPay'
          />
          <DFAlert i18nKey={null} type='success' headingLevel='h3' batches={[`cdcc-2`]}>
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation/you'
              condition='/onlyOneFilerHasCombatPay'
              batches={[`cdcc-2`]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation/you-and-your-spouse'
              condition='/bothFilersHasCombatPay'
              batches={[`cdcc-2`]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/cdcc-combat-pay-modal'
              batches={[`cdcc-0`, `cdcc-credits-0`]}
            />
          </DFAlert>

          <SaveAndOrContinueAndSetFactButton
            i18nKey='button.continueIncludeCombatPayCdcc'
            sourcePath='/cdccCombatPayRecommendation'
            destinationPath='/cdccCombatPayElection'
            conditions={[`/someFilerHasCombatPay`, { operator: `isFalse`, condition: `/bothFilersHasCombatPay` }]}
            batches={[`cdcc-3`]}
          />
          <SaveAndOrContinueAndSetFactButton
            i18nKey='button.continueIncludeBothCombatPayCdcc'
            sourcePath='/cdccCombatPayRecommendation'
            destinationPath='/cdccCombatPayElection'
            condition='/bothFilersHasCombatPay'
            batches={[`cdcc-3`]}
          />
          <InternalLink
            i18nKey='/info/cdcc-shared/combat-pay-change/another-option-link'
            route={`/flow/credits-and-deductions/credits/cdcc-combat-pay-change`}
            batches={[`cdcc-2`]}
          />
        </Screen>

        <Screen route={`cdcc-combat-pay-change`} routeAutomatically={false}>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/cdcc-combat-pay-change/primary-filer-has-combat-pay'
            batches={[`cdcc-0`, `cdcc-credits-0`]}
            condition='/onlyPrimaryFilerHasCombatPay'
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/cdcc-combat-pay-change/mfj-spouse-has-combat-pay'
            batches={[`cdcc-0`, `cdcc-credits-0`]}
            condition='/onlyMFJSpouseHasCombatPay'
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/cdcc-combat-pay-change/both-filers-has-combat-pay'
            batches={[`cdcc-0`, `cdcc-credits-0`]}
            condition='/bothFilersHasCombatPay'
          />
          <DFAlert i18nKey={null} type='success' headingLevel='h3' batches={[`cdcc-0`, `cdcc-credits-0`]}>
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation/you'
              condition='/onlyOneFilerHasCombatPay'
              batches={[`cdcc-2`]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation/you-and-your-spouse'
              condition='/bothFilersHasCombatPay'
              batches={[`cdcc-2`]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/cdcc-combat-pay-modal'
              batches={[`cdcc-0`, `cdcc-credits-0`]}
            />
          </DFAlert>

          <Enum path='/cdccCombatPayElection' batches={[`cdcc-3`]} />

          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </Gate>
    <Gate condition='/flowShowCdccQualifyingPersonsLoop'>
      <Screen route='taxable-benefits-distribution-ko' condition='/flowKnockoutBenefitsDistribution' isKnockout>
        <IconDisplay name='ErrorOutline' size={9} batches={[`cdcc-0`, `cdcc-credits-0`]} isCentered />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-knockout'
          batches={[`cdcc-0`, `cdcc-credits-0`]}
        />
        <InfoDisplay i18nKey='/info/knockout/taxable-benefits-distribution-ko' batches={[`cdcc-0`, `cdcc-credits-0`]} />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>

      {/* <!-- identify qp --> */}
      <Screen route='cdcc-identify-qp' condition='/cdccMaybeHasQualifyingExpenses'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          batches={[`cdcc-0`]}
        />
        <Heading
          batches={[`cdcc-1`]}
          i18nKey='/heading/credits-and-deductions/credits/cdcc-identify-qp-one'
          condition={{ operator: `isFalse`, condition: `/cdccMoreThanOneQualifyingPerson` }}
        />
        <Heading
          batches={[`cdcc-1`]}
          i18nKey='/heading/credits-and-deductions/credits/cdcc-identify-qp-several'
          condition='/cdccMoreThanOneQualifyingPerson'
        />
        <ConditionalList
          batches={[`cdcc-1`]}
          i18nKey='/info/credits-and-deductions/credits/cdcc/cdcc-identify-qp'
          items={[
            {
              itemKey: `qps`,
              collection: `/cdccQualifyingPeople`,
            },
            {
              itemKey: `filers`,
              collection: `/cdccQualifyingFilers`,
            },
          ]}
        />
        <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/credits-and-deductions/credits/cdcc-identify-qp-more-info' />
        <SaveAndOrContinueButton />
      </Screen>
      {makeCdccQualifyingPersonsLoop(`/cdccQualifyingPeople`, `/cdccQualifyingPeople`)}
      {makeCdccQualifyingPersonsLoop(`/cdccQualifyingFilers`, `/cdccQualifyingFilers`)}
      <Gate condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'>
        <SubSubcategory route='cdcc-special-income-rule' headingLevel='h3'>
          {earnedIncomeRuleKnockoutCheckFlow(
            cdccSubSectionNames.CREDITS,
            `/showSpecialEarnedIncomeKnockoutFlowInCredits`,
            [`cdcc-0`]
          )}
        </SubSubcategory>
      </Gate>
    </Gate>
    <SubSubcategory route='cdcc-qualified-yes' headingLevel='h3' editable={false}>
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.cdccOneQualified'
        conditions={[`/cdccQualified`, `/cdccHasExactlyOneQualifyingPerson`]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.cdccQualified'
        conditions={[`/cdccQualified`, { operator: `isFalse`, condition: `/cdccHasExactlyOneQualifyingPerson` }]}
      />

      <Screen route='cdcc-qualified' condition='/cdccQualified'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/cdcc/cdcc-qualified-people'
          condition='/cdccMoreThanOneQualifyingPerson'
          batches={[`cdcc-1`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/cdcc/cdcc-qualified-person'
          condition='/cdccHasExactlyOneQualifyingPerson'
          batches={[`cdcc-1`]}
        />
        <ConditionalList
          batches={[`cdcc-1`]}
          i18nKey='/info/credits-and-deductions/credits/cdcc/cdcc-qualified'
          items={[
            {
              itemKey: `qps`,
              collection: `/cdccQualifyingPeople`,
            },
            {
              itemKey: `filers`,
              collection: `/cdccQualifyingFilers`,
            },
          ]}
        />
        <InfoDisplay i18nKey='/info/credits-and-deductions/credits/cdcc/cdcc-qualified-info' batches={[`cdcc-2`]} />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <SubSubcategory route='cdcc-qualified-no' headingLevel='h3' editable={false}>
      <Assertion
        type={`info`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.cdccNotQualified'
        conditions={[`/maybeEligibleForCdcc`, { operator: `isFalse`, condition: `/cdccQualified` }]}
      />
      <Screen route='cdcc-not-qualified' condition={{ operator: `isFalse`, condition: `/cdccQualified` }}>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          batches={[`cdcc-0`]}
        />
        <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/cdcc/cdcc-not-qualified' />
        <ConditionalList
          i18nKey='/info/credits-and-deductions/credits/intro-no-credits/reasons'
          items={CdccDisqualifyingItems}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    {/* Care Providers Inner Collection Hub */}
    {makeCdccCareProvidersCollection(`cdcc-care-providers`, true, `/cdccQualified`)}
    {/* Non dependent qualifying tin/pin loop */}
    {CdccNonDependentQpTinPinLoop}
  </Gate>
);
