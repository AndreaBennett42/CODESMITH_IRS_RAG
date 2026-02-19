import { CollectionLoop, Gate, Screen, SubSubcategory, Assertion } from '../../flowDeclarations.js';
import {
  Boolean,
  ConditionalList,
  ContextHeading,
  DFModal,
  Dollar,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
} from '../../ContentDeclarations.js';
import { RawCondition } from '../../Condition.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';

type EdcKeys =
  | `taxLiability`
  | `agiAndFilingStatus`
  | `nonTaxablePaymentAndFilingStatus`
  | `ageAndDisability`
  | `ageAndDisabilityWhenMFJBoth`
  | `ageAndDisabilityWhenMFJSpouse`
  | `mandatoryRetirementAge`
  | `mandatoryRetirementAgeWhenMFJBoth`
  | `mandatoryRetirementAgeWhenMFJSpouse`
  | `taxableDisabilityIncome`
  | `taxableDisabilityIncomeWhenMFJBoth`
  | `taxableDisabilityIncomeWhenMFJSpouse`;

type EdcAssertionKeys = `filerQualifies` | `filerDoesntQualify`;

export const edcAssertionConditions: Record<EdcAssertionKeys, RawCondition[]> = {
  filerDoesntQualify: [{ operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` }],
  filerQualifies: [
    { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    `/qualifiedForCreditForElderlyAndDisabled`,
  ],
};
export const edcDisqualificationConditions: Record<EdcKeys, RawCondition[]> = {
  // TODO: Remove this first item. It should be unreachable now that the EDC section is skipped if there is no tax
  // liability. Chose to leave in place 12/26/24 to avoid the possibility of introducing a late regression. - CG
  taxLiability: [`/isDisqualifiedFromEdcBasedOnTaxLiability`],
  agiAndFilingStatus: [`/isDisqualifiedFromEdcBasedOnAgiLimit`],
  nonTaxablePaymentAndFilingStatus: [`/isDisqualifiedFromEdcBasedOnNonTaxablePayments`],
  ageAndDisability: [
    `/primaryFiler/isDisqualifedFromEdcBasedOnAgeOrDisability`,
    { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/isDisqualifedFromEdcBasedOnAgeOrDisability` },
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  ageAndDisabilityWhenMFJSpouse: [
    { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/isDisqualifedFromEdcBasedOnAgeOrDisability` },
    `/secondaryFiler/isDisqualifedFromEdcBasedOnAgeOrDisability`,
    `/isFilingStatusMFJ`,
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  ageAndDisabilityWhenMFJBoth: [
    `/primaryFiler/isDisqualifedFromEdcBasedOnAgeOrDisability`,
    `/secondaryFiler/isDisqualifedFromEdcBasedOnAgeOrDisability`,
    `/isFilingStatusMFJ`,
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  mandatoryRetirementAge: [
    `/primaryFiler/isDisqualifiedByMandatoryRetirementAge`,
    { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/isDisqualifiedByMandatoryRetirementAge` },
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  mandatoryRetirementAgeWhenMFJSpouse: [
    { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/isDisqualifiedByMandatoryRetirementAge` },
    { condition: `/secondaryFiler/isDisqualifiedByMandatoryRetirementAge` },
    `/isFilingStatusMFJ`,
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  mandatoryRetirementAgeWhenMFJBoth: [
    { condition: `/primaryFiler/isDisqualifiedByMandatoryRetirementAge` },
    { condition: `/secondaryFiler/isDisqualifiedByMandatoryRetirementAge` },
    `/isFilingStatusMFJ`,
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  taxableDisabilityIncome: [
    `/primaryFiler/isDisqualifiedFromEdcBasedOnTaxableDisabilityIncome`,
    {
      operator: `isFalseOrIncomplete`,
      condition: `/secondaryFiler/isDisqualifiedFromEdcBasedOnTaxableDisabilityIncome`,
    },
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  taxableDisabilityIncomeWhenMFJSpouse: [
    { operator: `isFalseOrIncomplete`, condition: `/primaryFiler/isDisqualifiedFromEdcBasedOnTaxableDisabilityIncome` },
    `/secondaryFiler/isDisqualifiedFromEdcBasedOnTaxableDisabilityIncome`,
    `/isFilingStatusMFJ`,
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
  taxableDisabilityIncomeWhenMFJBoth: [
    `/primaryFiler/isDisqualifiedFromEdcBasedOnTaxableDisabilityIncome`,
    `/secondaryFiler/isDisqualifiedFromEdcBasedOnTaxableDisabilityIncome`,
    `/isFilingStatusMFJ`,
    { operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` },
  ],
};

export const EdcDisqualifyingItems: ItemConfig[] = [
  // These two items skip the EDC section, and so are only included here, and not on the `edc-not-qualified` screen
  {
    itemKey: `subListEdc-noTaxLiability`,
    conditions: [
      { operator: `isFalse`, condition: `/hasTaxLiabilityBeforeNrCredits` },
      `/maybeEligibleForCreditForElderlyAndDisabledWoTaxLiability`,
    ],
  },
  {
    itemKey: `subListEdc-reachedCreditLimit`,
    conditions: [
      `/hasTaxLiabilityBeforeNrCredits`,
      { operator: `isFalse`, condition: `/maybeEligibleForCreditForElderlyAndDisabled` },
      `/maybeEligibleForCreditForElderlyAndDisabledWoTaxLiability`,
    ],
  },
  {
    itemKey: `subListEdc-agiAndFilingStatus`,
    conditions: edcDisqualificationConditions.agiAndFilingStatus,
  },
  {
    itemKey: `subListEdc-nonTaxablePaymentAndFilingStatus`,
    conditions: edcDisqualificationConditions.nonTaxablePaymentAndFilingStatus,
  },
  {
    itemKey: `subListEdc-ageAndDisability`,
    conditions: edcDisqualificationConditions.ageAndDisability,
  },
  {
    itemKey: `subListEdc-ageAndDisabilityWhenMFJSpouse`,
    conditions: edcDisqualificationConditions.ageAndDisabilityWhenMFJSpouse,
  },
  {
    itemKey: `subListEdc-ageAndDisabilityWhenMFJBoth`,
    conditions: edcDisqualificationConditions.ageAndDisabilityWhenMFJBoth,
  },
  {
    itemKey: `subListEdc-mandatoryRetirementAge`,
    conditions: edcDisqualificationConditions.mandatoryRetirementAge,
  },
  {
    itemKey: `subListEdc-mandatoryRetirementAgeWhenMFJSpouse`,
    conditions: edcDisqualificationConditions.mandatoryRetirementAgeWhenMFJSpouse,
  },
  {
    itemKey: `subListEdc-mandatoryRetirementAgeWhenMFJBoth`,
    conditions: edcDisqualificationConditions.mandatoryRetirementAgeWhenMFJBoth,
  },
  {
    itemKey: `subListEdc-taxableDisabilityIncome`,
    conditions: edcDisqualificationConditions.taxableDisabilityIncome,
  },
  {
    itemKey: `subListEdc-taxableDisabilityIncomeWhenMFJSpouse`,
    conditions: edcDisqualificationConditions.taxableDisabilityIncomeWhenMFJSpouse,
  },
  {
    itemKey: `subListEdc-taxableDisabilityIncomeWhenMFJBoth`,
    conditions: edcDisqualificationConditions.taxableDisabilityIncomeWhenMFJBoth,
  },
];

export const EdcSubSubcategory = (
  <Gate condition='/maybeEligibleForCreditForElderlyAndDisabled'>
    <Screen route='edc-breather'>
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/edc' batches={[`edc-0`]} />
      <Heading
        i18nKey='/heading/credits-and-deductions/credits/edc-breather'
        batches={[`information-architecture-0`]}
      />
      <SaveAndOrContinueButton />
    </Screen>
    <Gate condition='/maybeEligibleForEdcBasedOnDisability'>
      <CollectionLoop
        collection='/filersMaybeEligibleForDisability'
        loopName='/filersMaybeEligibleForDisability'
        autoIterate={true}
        collectionItemCompletedCondition='/filers/*/initialDisabilityIsComplete'
      >
        {/* TODO if more than one item in this loop we need to adjust this heading 
        as it will appear with context header twice in dataview. Not applicable 2024 */}
        <SubSubcategory route='disability' headingLevel='h3' borderStyle='auto'>
          <Screen route='edc-disability-intro-screen'>
            <ContextHeading i18nKey='/heading/credits-and-deductions/credits/edc' batches={[`edc-0`]} />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-disability-intro-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-disability-intro-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='edc-permanent-total-disability'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/edc'
              batches={[`edc-0`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-permanent-total-disability-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-permanent-total-disability-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/edc-permanent-total-disability-snack'
              batches={[`edc-0`]}
            />
            <Boolean path='/filers/*/isRetOnPermOrTotalDisability' batches={[`edc-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='edc-mandatory-retirement-age'
            condition={{ operator: `isTrue`, condition: `/filers/*/isRetOnPermOrTotalDisability` }}
          >
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/edc'
              batches={[`edc-0`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-mandatory-retirement-age-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-mandatory-retirement-age-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/edc-mandatory-retirement-age-info'
              batches={[`edc-0`]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/edc-mandatory-retirement-age-snack'
              batches={[`edc-0`]}
            />
            <Boolean path='/filers/*/employerHasMandatoryRetirementAge' batches={[`edc-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='edc-mandatory-retirement-age-reached'
            condition='/filers/*/employerHasMandatoryRetirementAgeAndRetired'
          >
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/edc'
              batches={[`edc-0`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-mandatory-retirement-age-reached-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-mandatory-retirement-age-reached-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <Boolean path='/filers/*/hasMetEmployerMandatoryRetirementAge' batches={[`edc-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='edc-disability-income' condition={`/filers/*/hasDisabilityIncome`}>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/edc'
              batches={[`edc-0`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-disability-income-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-disability-income-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/edc-disability-income-wages-and-disability-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/edc-disability-income-wages-and-disability-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/edc-disability-income-other-sources-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/edc-disability-income-other-sources-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <DFModal i18nKey='/info/credits-and-deductions/credits/edc-disability-income-snack' batches={[`edc-0`]} />
            <Boolean path='/filers/*/arePaymentsTaxDisabilityIncome' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='edc-disability-income-amount'
            condition={{
              operator: `isTrue`,
              condition: `/filers/*/hasTaxableDisabilityPaymentsAndRetOnDisability`,
            }}
          >
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/edc'
              batches={[`edc-0`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-disability-income-amount-primary-only'
              batches={[`edc-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/edc-disability-income-amount-spouse-only'
              batches={[`edc-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/edc-disability-income-amount-snack-primary'
              batches={[`edc-0`]}
              condition={`/filers/*/isPrimaryFiler`}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/edc-disability-income-amount-snack-spouse'
              batches={[`edc-0`]}
              condition={`/filers/*/isSecondaryFiler`}
            />
            <Dollar path='/filers/*/writableTotalTaxableDisabilityAmount' batches={[`edc-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </CollectionLoop>
    </Gate>
    <SubSubcategory route='qualified-for-edc' headingLevel='h3' borderStyle='auto'>
      <Screen
        route='edc-nontaxable-payments'
        condition={{ operator: `isTrue`, condition: `/tpCanBeAskedAboutNonTaxablePayments` }}
      >
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/edc'
          batches={[`edc-0`]}
        />
        <ContextHeading
          displayOnlyOn='data-view'
          i18nKey='/heading/credits-and-deductions/credits/edc'
          batches={[`edc-0`]}
          condition={{ operator: `isFalse`, condition: `/maybeEligibleForEdcBasedOnDisability` }}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/edc-nontaxable-payments'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`edc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/edc-nontaxable-payments-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`edc-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/edc-nontaxable-payments-only-use'
          batches={[`edc-0`]}
        />
        <DFModal i18nKey='/info/credits-and-deductions/credits/edc-nontaxable-payments' batches={[`edc-0`]} />
        <Boolean path='/writableHasSelfReportedNonTaxablePayments' batches={[`edc-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='edc-add-nontaxable-payment' condition='/hasSelfReportedNonTaxablePayments'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/edc'
          batches={[`edc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/edc-nontaxable-payments-yes'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`edc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/edc-nontaxable-payments-yes-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`edc-0`]}
        />
        <DFModal i18nKey='/info/credits-and-deductions/credits/edc-nontaxable-payments' batches={[`edc-0`]} />
        <Dollar path='/writableEdcSelfReportedNonTaxablePaymentAmount' batches={[`edc-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <SubSubcategory route='edc-not-qualified-outcome' editable={false} headingLevel='h2'>
        <Assertion
          type={`info`}
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.filerDoesntQualifyForEDC'
          conditions={[
            ...edcAssertionConditions.filerDoesntQualify,
            `/maybeEligibleForCreditForElderlyAndDisabled`,
            `/edcShouldShowCreditOutcomes`,
          ]}
        />
        <Screen
          route='edc-not-qualified'
          condition={{ operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` }}
        >
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/edc'
            batches={[`edc-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/edc-not-qualified'
            batches={[`information-architecture-0`]}
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/edc-not-qualified-reasons' batches={[`edc-0`]} />
          <ConditionalList
            i18nKey='/info/credits-and-deductions/credits/edc-not-qualified-reasons'
            batches={[`edc-0`]}
            items={[
              {
                itemKey: `tax-liability`,
                conditions: edcDisqualificationConditions.taxLiability,
              },
              {
                itemKey: `filing-status-and-agi`,
                conditions: edcDisqualificationConditions.agiAndFilingStatus,
              },
              {
                itemKey: `filing-status-and-nontaxable-payments`,
                conditions: edcDisqualificationConditions.nonTaxablePaymentAndFilingStatus,
              },
              {
                itemKey: `age-and-disability`,
                conditions: edcDisqualificationConditions.ageAndDisability,
              },
              {
                itemKey: `age-and-disability-mfj-spouse`,
                conditions: edcDisqualificationConditions.ageAndDisabilityWhenMFJSpouse,
              },
              {
                itemKey: `age-and-disability-mfj-both`,
                conditions: edcDisqualificationConditions.ageAndDisabilityWhenMFJBoth,
              },
              {
                itemKey: `mandatory-retirement-age`,
                conditions: edcDisqualificationConditions.mandatoryRetirementAge,
              },
              {
                itemKey: `mandatory-retirement-age-mfj-spouse`,
                conditions: edcDisqualificationConditions.mandatoryRetirementAgeWhenMFJSpouse,
              },
              {
                itemKey: `mandatory-retirement-age-mfj-both`,
                conditions: edcDisqualificationConditions.mandatoryRetirementAgeWhenMFJBoth,
              },
              {
                itemKey: `taxable-disability-income`,
                conditions: edcDisqualificationConditions.taxableDisabilityIncome,
              },
              {
                itemKey: `taxable-disability-income-mfj-spouse`,
                conditions: edcDisqualificationConditions.taxableDisabilityIncomeWhenMFJSpouse,
              },
              {
                itemKey: `taxable-disability-income-mfj-both`,
                conditions: edcDisqualificationConditions.taxableDisabilityIncomeWhenMFJBoth,
              },
            ]}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='edc-qualified-outcome' editable={false} headingLevel='h3'>
        <Assertion
          type={`success`}
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.filerQualifiedForEDC'
          conditions={[...edcAssertionConditions.filerQualifies, `/edcShouldShowCreditOutcomes`]}
        />
        <Screen
          route='edc-qualified'
          condition={{ operator: `isTrue`, condition: `/qualifiedForCreditForElderlyAndDisabled` }}
        >
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/edc'
            batches={[`edc-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/edc-qualified'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            batches={[`information-architecture-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/edc-qualified-mfj'
            condition='/isFilingStatusMFJ'
            batches={[`information-architecture-0`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/edc-qualified-let-you-know'
            batches={[`information-architecture-0`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/edc-qualified-under-disability'
            conditions={[
              { operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/isMaybeQualifiedforEdcThroughDisability` },
              { condition: `/primaryFiler/isMaybeQualifiedforEdcThroughDisability` },
            ]}
            batches={[`edc-0`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/edc-qualified-under-disability-spouse-only'
            conditions={[
              { condition: `/secondaryFiler/isMaybeQualifiedforEdcThroughDisability` },
              { operator: `isFalse`, condition: `/primaryFiler/isMaybeQualifiedforEdcThroughDisability` },
              { condition: `/isFilingStatusMFJ` },
            ]}
            batches={[`edc-0`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/edc-qualified-under-disability-both'
            conditions={[
              { condition: `/primaryFiler/isMaybeQualifiedforEdcThroughDisability` },
              { condition: `/secondaryFiler/isMaybeQualifiedforEdcThroughDisability` },
              { condition: `/isFilingStatusMFJ` },
            ]}
            batches={[`edc-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Gate condition='/qualifiedForCreditForElderlyAndDisabled'>
        <CollectionLoop
          collection='/filersQualifiedForEdcThroughDisability'
          loopName='/filersQualifiedForEdcThroughDisability'
          autoIterate={true}
          collectionItemCompletedCondition='/filers/*/hasCompletedPhysicianStatementQuestions'
        >
          <SubSubcategory route='edc-physician-statement' headingLevel='h3'>
            <Screen route='edc-physician-statement-previous-with-line-b'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/edc'
                batches={[`edc-0`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-line-b-primary'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-line-b-secondary'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-line-b-primary'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-line-b-primary-spouse'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <Boolean path='/filers/*/hasPhysicianStatementBothDisabledAndWillNotImprove' batches={[`edc-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='edc-physician-statement-1983'
              condition={{
                operator: `isFalse`,
                condition: `/filers/*/hasPhysicianStatementBothDisabledAndWillNotImprove`,
              }}
            >
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/edc'
                batches={[`edc-0`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-1983-primary'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-1983-secondary'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <Boolean path='/filers/*/hasPhysicianStatementBefore1983' batches={[`edc-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='edc-physician-statement-self-certify'
              condition='/filers/*/askAboutPhysicianStatementSelfCertify'
            >
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/edc'
                batches={[`edc-0`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-self-certify'
                batches={[`edc-0`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-self-certify-primary'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-self-certify-secondary'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <Boolean path='/filers/*/hasSelfCertPhysStatmntOrVetDisbltyVerified' batches={[`edc-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='edc-physician-statement-needed' condition='/filers/*/physicianStatementNeeded'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/edc'
                batches={[`edc-0`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-needed-primary'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-needed-secondary'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-needed'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-needed-spouse'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='edc-physician-statement-not-needed' condition='/filers/*/physicianStatementNotNeeded'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/edc'
                batches={[`edc-0`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-not-needed-primary'
                batches={[`edc-0`]}
                condition='/filers/*/isPrimaryFiler'
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/edc-physicians-statement-not-needed-secondary'
                batches={[`edc-0`]}
                condition='/filers/*/isSecondaryFiler'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/edc-physicians-statement-not-needed'
                batches={[`edc-0`]}
              />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
        </CollectionLoop>
      </Gate>
    </SubSubcategory>
  </Gate>
);
