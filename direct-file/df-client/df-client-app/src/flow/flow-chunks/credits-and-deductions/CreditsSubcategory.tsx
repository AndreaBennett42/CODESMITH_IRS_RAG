/* eslint-disable max-len */
import { Assertion, CollectionLoop, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ConditionalList,
  ContextHeading,
  DatePicker,
  DFModal,
  DFAlert,
  Dollar,
  Enum,
  Heading,
  HelpLink,
  IconDisplay,
  InfoDisplay,
  InternalLink,
  SaveAndOrContinueButton,
  SetFactAction,
  SummaryTable,
  DFAccordion,
  ConditionalAccordion,
  KnockoutButton,
  CollectionItemManager,
  LimitingString,
  MultiEnum,
  GenericString,
  TaxReturnAlert,
  MefAlert,
} from '../../ContentDeclarations.js';
import { SummaryListItemConfig } from '../../../components/SummaryTable/SummaryTable.js';
import { EdcSubSubcategory, EdcDisqualifyingItems } from './EdcSubSubcategory.js';
import { SaversCreditSubSubcategory, SaversCreditDisqualifyingItems } from './SaversCreditSubSubcategory.js';
import { CtcOdcSubSubcategory, CtcDisqualifyingItems, OdcDisqualifyingItems } from './CtcOdcSubSubcategory.js';
import { CdccSubSubcategory, CdccDisqualifyingItems } from './CdccSubSubcategory.js';
import { EitcSubSubcategory, EitcDisqualifyingItems } from './EitcSubSubcategory.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';

const PtcDisqualifyingItems: ItemConfig[] = [
  {
    itemKey: `subListPtc-depTpIsClaimedEnrolledSelfAndNotOthers`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcDependentTpDidNotEnrollSelfOrOthers`,
    ],
  },
  {
    itemKey: `subListPtc-depTpDidNotEnrollSelfOrOthers`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcDependentTpIsClaimedDidEnrollSelfAndNotOthers`,
    ],
  },
  {
    itemKey: `subListPtc-depTpDidNotEnrollSelfAndEnrolledOthersWhoIsPartOfDiffTaxFam`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcDependentTpDidNotEnrollSelfAndEnrolledOthersWhoIsPartOfDifferentTaxFamily`,
    ],
  },
  {
    itemKey: `subListPtc-depTpIsClaimedEnrolledSelfAndOthersWhoIsPartOfDiffTaxFam`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcDependentTpIsClaimedDidEnrollSelfAndOthersAndPartOfDifferentTaxFamily`,
    ],
  },
  {
    itemKey: `subListPtc-noQualifiedPlanSelfNoDep`,
    conditions: [
      `/noQualifiedPlanAndNotADependentTaxPayer`,
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
      { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
    ],
  },
  {
    itemKey: `subListPtc-noQualifiedPlanSelfHasDep`,
    conditions: [
      `/noQualifiedPlanAndNotADependentTaxPayer`,
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
      `/claimedDependentsCount`,
    ],
  },
  {
    itemKey: `subListPtc-noQualifiedPlanMfjNoDep`,
    conditions: [
      `/noQualifiedPlanAndNotADependentTaxPayer`,
      `/isFilingStatusMFJ`,
      { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
    ],
  },
  {
    itemKey: `subListPtc-noQualifiedPlanMfjHasDep`,
    conditions: [`/noQualifiedPlanAndNotADependentTaxPayer`, `/isFilingStatusMFJ`, `/claimedDependentsCount`],
  },
  {
    itemKey: `subListPtc-hohConsidererdUnmarried`,
    conditions: [
      { operator: `isFalse`, condition: `/noQualifiedPlanAndNotADependentTaxPayer` },
      // Main condition
      {
        operator: `isFalse`,
        condition: `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      },
      `/filerNotQualifiedForPtcBecauseOfStatusAndResidenceInfo`,
    ],
  },
  {
    itemKey: `subListPtc-depTpNotClaimedOnlyEnrolledSelf`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcNotClaimedDependentTpWhoOnlyEnrolledSelf`,
    ],
  },
  {
    itemKey: `subListPtc-depTpNotClaimedOnlyEnrolledSomeoneElseNotPartOfDiffTaxFam`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcNotClaimedDependentTpWhoOnlyEnrolledSomeoneElseNotPartOfDiffTaxFam`,
    ],
  },
  {
    itemKey: `subListPtc-depTpNotClaimedOnlyEnrolledSelfAndSomeoneElseNotPartOfDiffTaxFam`,
    conditions: [
      `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      `/ptcNotClaimedDependentTpWhoEnrolledSelfAndSomeoneElseNotPartOfDiffTaxFam`,
    ],
  },
  {
    itemKey: `subListPtc-mfsExceptionNotApplicable`,
    conditions: [
      { operator: `isFalse`, condition: `/noQualifiedPlanAndNotADependentTaxPayer` },
      { operator: `isFalse`, condition: `/filerNotQualifiedForPtcBecauseOfStatusAndResidenceInfo` },
      // Main condition
      {
        operator: `isFalse`,
        condition: `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
      },
      `/notPtcApplicableTaxPayerSoFar`,
    ],
  },
  {
    itemKey: `subListPtc-below100Fpl`,
    conditions: [
      { operator: `isFalse`, condition: `/noQualifiedPlanAndNotADependentTaxPayer` },
      { operator: `isFalse`, condition: `/filerNotQualifiedForPtcBecauseOfStatusAndResidenceInfo` },
      // Main condition
      { operator: `isFalse`, condition: `/below100ThresholdAndNotEligibleForMedicaidBecauseOfImmigrationStatus` },
    ],
  },
  {
    itemKey: `subListPtc-noCoverageMonths`,
    conditions: [
      { operator: `isFalse`, condition: `/noQualifiedPlanAndNotADependentTaxPayer` },
      { operator: `isFalse`, condition: `/filerNotQualifiedForPtcBecauseOfStatusAndResidenceInfo` },
      // Main condition
      `/potentiallyApplicableTaxPayer`,
      `/ptcHasZeroCoverageMonths`,
    ],
  },
];

const CreditsDisqualificationAccordion = (
  <ConditionalAccordion
    i18nKey='/info/credits-and-deductions/credits/intro-no-credits/reasons'
    batches={[`savers-8880-0`, `credits-ia-1`]}
    conditions={[{ operator: `isFalse`, condition: `/isMFJDependent` }, `/hasBeenDisqualifiedForAtLeastOneCredit`]}
    items={[
      ...PtcDisqualifyingItems,
      ...CdccDisqualifyingItems,
      ...EdcDisqualifyingItems,
      ...SaversCreditDisqualifyingItems,
      ...CtcDisqualifyingItems,
      ...OdcDisqualifyingItems,
      ...EitcDisqualifyingItems,
    ]}
  />
);

// This table will be displayed if the user has refundable credits.
// The fact to display this table is `/qualifiedForRefundableCredit`.
// If you change this table, you must also change that fact.
const CreditsSummaryRefundableItems: SummaryListItemConfig[] = [
  {
    itemKey: `refundableCreditsHeader`,
  },
  {
    itemKey: `ptcZeroOrPositive`,
    indent: true,
    conditions: [`/needsToFileForm8962`, `/netPtcAmountIsZeroOrPositive`],
  },
  {
    itemKey: `ptcNegative`,
    indent: true,
    conditions: [`/needsToFileForm8962`, `/netPtcAmountIsNegative`],
  },
  {
    itemKey: `actc`,
    indent: true,
    conditions: [`/ctcQualified`],
  },
  {
    itemKey: `eitc`,
    indent: true,
    conditions: [`/eitcQualified`],
  },
  {
    itemKey: `total`,
    showTopBorder: true,
  },
];

// This table will be displayed if the user has nonrefundable credits.
// The fact to display this table is `/qualifiedForNonrefundableCredit`.
// If you change this table, you must also change that fact.
const CreditsSummaryNonRefundableItems: SummaryListItemConfig[] = [
  {
    itemKey: `nonrefundableCreditsHeader`,
  },
  {
    itemKey: `cdcc`,
    indent: true,
    conditions: [`/isReceivingCdccCredit`],
  },
  {
    itemKey: `edc`,
    indent: true,
    conditions: [`/qualifiedForCreditForElderlyAndDisabled`],
  },
  {
    itemKey: `saversNotMFJ`,
    indent: true,
    conditions: [`/qualifiedForSaverCredit`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }],
  },
  {
    itemKey: `saversMFJSinglePayerQualified`,
    indent: true,
    conditions: [
      `/qualifiedForSaverCredit`,
      `/isFilingStatusMFJ`,
      { operator: `isComplete`, condition: `/nameOfOnlyQualifier` },
    ],
  },
  {
    itemKey: `saversMFJBothQualified`,
    indent: true,
    conditions: [
      `/qualifiedForSaverCredit`,
      `/isFilingStatusMFJ`,
      { operator: `isIncomplete`, condition: `/nameOfOnlyQualifier` },
    ],
  },
  {
    itemKey: `odc`,
    indent: true,
    conditions: [`/odcQualified`],
  },
  {
    itemKey: `ctc`,
    indent: true,
    conditions: [`/ctcQualified`],
  },
  {
    itemKey: `total`,
    showTopBorder: true,
  },
];

export const CreditsSubcategory = (
  <Subcategory
    route='credits'
    completeIf='/creditsSectionComplete'
    dataItems={[
      {
        itemKey: `refundableCredits`,
        conditions: [`/hasRefundableCredits`],
      },
      {
        itemKey: `nonRefundableCredits`,
        conditions: [`/hasNonRefundableCredits`],
      },
    ]}
  >
    <Gate condition='/hasPotentialCredits'>
      {/* Credits intro */}
      <Screen route='credits-intro-potential-credits'>
        <ContextHeading i18nKey='/heading/credits-and-deductions/credits/credits' />
        <Heading i18nKey='/heading/credits-and-deductions/credits/intro-potential-credits' batches={[`cdcc-2`]} />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/intro-potential-credits'
          batches={[`savers-8880-0`, `edc-0`, `cdcc-2`, `ptc-2`]}
        />
        <SaveAndOrContinueButton />
      </Screen>

      {/* Premium tax credit (PTC) */}
      <Gate condition='/maybeEligibleForPtc'>
        <Screen route='ptc-intro'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/ptc'
            batches={[`ptc-1`]}
          />
          <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-intro' batches={[`ptc-1`]} />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/ptc-intro1-with-state-marketplace'
            condition={`/isResidentOfStateWithMarketplace`}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/ptc-intro1-without-state-marketplace'
            condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-intro2' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='marketplace-breather'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ptc' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-marketplace-breather' />
          <DFModal
            i18nKey='/info/credits-and-deductions/credits/ptc-marketplace-breather-with-state-plan'
            condition={`/isResidentOfStateWithMarketplace`}
          />
          <DFModal
            i18nKey='/info/credits-and-deductions/credits/ptc-marketplace-breather-without-state-plan'
            condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
          />
          <DFModal
            i18nKey='/info/credits-and-deductions/credits/check-marketplace-plan-for-ptc-with-state-plan'
            condition={`/isResidentOfStateWithMarketplace`}
          />
          <DFModal
            i18nKey='/info/credits-and-deductions/credits/check-marketplace-plan-for-ptc-without-state-plan'
            condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
          />
          <SetFactAction
            condition={{ condition: `data-import`, section: `has-at-least-one-1095-a` }}
            source={`/flowTrue`}
            path={`/has1095AInIRDAS`}
          />
          <SetFactAction
            condition={{ condition: `data-import`, section: `has-no-1095-a` }}
            source={`/flowFalse`}
            path={`/has1095AInIRDAS`}
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition={{ operator: `isFalse`, condition: `/isMFJDependent` }}>
          <Screen route='1095-import-info' condition={`/has1095AInIRDAS`}>
            <Heading i18nKey='/heading/credits-and-deductions/credits/1095-import-info' />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-single-no-dependents'
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/claimedDependentsCount` },
              ]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-single-dependents'
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { condition: `/claimedDependentsCount` },
              ]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-mfj-no-dependents'
              conditions={[
                { condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/claimedDependentsCount` },
              ]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-mfj-dependents'
              conditions={[{ condition: `/isFilingStatusMFJ` }, { condition: `/claimedDependentsCount` }]}
            />
            <DFAlert
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-alert-no-dependents'
              type={`info`}
              headingLevel={`h3`}
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/claimedDependentsCount` },
              ]}
            />
            <DFAlert
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-alert-mfj-or-dependents'
              type={`info`}
              headingLevel={`h3`}
              condition={`/isFilingStatusMFJ`}
            />
            <DFModal i18nKey='/info/credits-and-deductions/credits/received-1095-a-without-marketplace-plan' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1095-import-info-no-form' condition={{ operator: `isFalse`, condition: `/has1095AInIRDAS` }}>
            <Heading i18nKey='/heading/credits-and-deductions/credits/1095-import-info-no-form' />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-intro' />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-mfj-no-deps'
              conditions={[
                { condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/claimedDependentsCount` },
              ]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-mfj-deps'
              conditions={[{ condition: `/isFilingStatusMFJ` }, { condition: `/claimedDependentsCount` }]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-single-deps'
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { condition: `/claimedDependentsCount` },
              ]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-confirm-single-no-deps'
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/claimedDependentsCount` },
              ]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-confirm-single-deps'
              conditions={[
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { condition: `/claimedDependentsCount` },
              ]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-confirm-mfj-no-deps'
              conditions={[
                { condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/claimedDependentsCount` },
              ]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/1095-import-info-no-form-confirm-mfj-deps'
              conditions={[{ condition: `/isFilingStatusMFJ` }, { condition: `/claimedDependentsCount` }]}
            />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
        <SubSubcategory route='ptc-eligibility' headingLevel='h3'>
          <Gate condition='/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits'>
            <Screen route='dep-tp-self-enrollment'>
              <ContextHeading i18nKey='/heading/credits-and-deductions/credits/ptc' batches={[`ptc-1`]} />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-dep-tp-self-enrollment-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-dep-tp-self-enrollment-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-self-enrollment-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-self-enrollment-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-self-enrollment'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <Boolean path='/dependentTpSelfEnrolled' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='dep-tp-enrolled-someone-else'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-dep-tp-enrolled-others'
                batches={[`ptc-1`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-enrolled-others'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/what-is-qualified-marketplace-plan'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <Boolean path='/dependentTpEnrolledSomeoneElse' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='dep-tp-enrollee-not-part-of-tax-family' condition='/dependentTpEnrolledSomeoneElse'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-dep-tp-enrollee-not-part-of-tax-family'
                batches={[`ptc-1`]}
              />
              <DFModal i18nKey='/info/credits-and-deductions/credits/what-is-tax-family' batches={[`ptc-1`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-enrollee-not-part-of-tax-family'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/what-if-enrolled-more-than-one'
                batches={[`ptc-1`]}
              />
              <Boolean path='/writableDependentTpOtherMembersArePartOfDifferentTaxFamily' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='dep-tp-ko' condition='/dependentTpRequiresAllocations' isKnockout>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-ko-someone-belongs-to-different-tax-family'
                batches={[`ptc-1`]}
                condition='/notClaimedAndPartOfDifferentTaxFamily'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-ko-someone-can-be-claimed-and-enrolled-self'
                batches={[`ptc-1`]}
                condition='/claimedAndNotPartOfDifferentTaxFamily'
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-ko-shared' batches={[`ptc-1`]} />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen route='dep-tp-ptc-keep-going' condition='/dependentTpNeedToCollectMoreInfo'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-dep-tp-keep-going' batches={[`ptc-1`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-did-not-claimed-and-only-enrolled-self'
                batches={[`ptc-2`]}
                condition='/ptcDependentTpNotClaimedAndOnlyEnrolledSelf'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-did-only-enrolled-others-and-not-part-of-different-tax-family'
                batches={[`ptc-1`]}
                condition='/ptcDependentTpOnlyEnrolledOthersAndNotPartOfDifferentTaxFamily'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-is-not-claimed-did-enroll-self-and-others-and-not-part-of-diff-tax-family'
                batches={[`ptc-1`]}
                condition='/ptcDependentTpNotClaimedEnrolledSelfAndOthersNotPartOfDifferentTaxFamily'
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-keep-going' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Gate
            condition={{
              operator: `isFalse`,
              condition: `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
            }}
          >
            <Screen route='qualified-marketplace-plan'>
              <MefAlert mefErrorCode='F8962-070' i18nKey='family-and-household' type='warning' />
              <ContextHeading i18nKey='/heading/credits-and-deductions/credits/ptc' batches={[`ptc-1`]} />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-self-no-deps-without-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-self-with-deps-without-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  `/claimedDependentsCount`,
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-no-deps-without-state-plan'
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-with-deps-without-state-plan'
                conditions={[
                  `/isFilingStatusMFJ`,
                  `/claimedDependentsCount`,
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-self-no-deps-with-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  `/isResidentOfStateWithMarketplace`,
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-self-with-deps-with-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  `/claimedDependentsCount`,
                  `/isResidentOfStateWithMarketplace`,
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-no-deps-with-state-plan'
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  `/isResidentOfStateWithMarketplace`,
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-with-deps-with-state-plan'
                conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`, `/isResidentOfStateWithMarketplace`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-with-deps-with-state-plan'
                condition='/isResidentOfStateWithMarketplace'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-with-deps-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-qualified-marketplace-plan-mfj-with-deps' />
              <DFAlert
                i18nKey={`/info/credits-and-deductions/credits/data-import-1095-a-single`}
                type={`info`}
                headingLevel={`h3`}
                conditions={[
                  { condition: `/has1095AInIRDAS` },
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                ]}
              />
              <DFAlert
                i18nKey={`/info/credits-and-deductions/credits/data-import-1095-a-mfj`}
                type={`info`}
                headingLevel={`h3`}
                conditions={[`/has1095AInIRDAS`, `/isFilingStatusMFJ`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/check-marketplace-plan-for-ptc-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/check-marketplace-plan-for-ptc-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <DFModal i18nKey='/info/credits-and-deductions/credits/what-if-enrolled-someone-else' />
              <Boolean path='/writableHasPtcQualifyingPlan' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Gate condition='/hasPtcQualifyingPlan'>
            <Screen route='qsehra-ko' condition='/hasPtcQualifyingPlanAndQsehra' isKnockout>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/qsehra-ko-self'
                batches={[`ptc-1`]}
                condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/qsehra-ko-mfj'
                batches={[`ptc-1`]}
                condition='/isFilingStatusMFJ'
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/qsehra-ko-shared' batches={[`ptc-1`]} />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen route='dependents-filing-requirement' condition='/claimedDependentsCount'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-dependents-filing-req'
                batches={[`ptc-1`]}
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-dependents-required' batches={[`ptc-1`]} />
              <Boolean path='/writableDependentsRequiredToFile' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='dependents-filing-req-ko' condition='/dependentsRequiredToFile' isKnockout>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/dependents-filing-req-ko'
                batches={[`ptc-1`]}
              />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen route='1095-a-check'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-self-no-deps-without-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-self-with-deps-without-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  `/claimedDependentsCount`,
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-mfj-no-deps-without-state-plan'
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-mfj-with-deps-without-state-plan'
                conditions={[
                  `/isFilingStatusMFJ`,
                  `/claimedDependentsCount`,
                  { operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` },
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-self-no-deps-with-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  `/isResidentOfStateWithMarketplace`,
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-self-with-deps-with-state-plan'
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  `/claimedDependentsCount`,
                  `/isResidentOfStateWithMarketplace`,
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-mfj-no-deps-with-state-plan'
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  `/isResidentOfStateWithMarketplace`,
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-1095-a-mfj-with-deps-with-state-plan'
                conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`, `/isResidentOfStateWithMarketplace`]}
              />
              <DFAlert
                i18nKey={`/info/credits-and-deductions/credits/data-import-get-1095-a-single`}
                type={`info`}
                headingLevel={`h3`}
                conditions={[`/has1095AInIRDAS`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
              />
              <DFAlert
                i18nKey={`/info/credits-and-deductions/credits/data-import-get-1095-a-mfj`}
                type={`info`}
                headingLevel={`h3`}
                conditions={[`/has1095AInIRDAS`, { condition: `/isFilingStatusMFJ` }]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-1095-a-check-self-no-deps'
                batches={[`ptc-3`]}
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-1095-a-check-self-with-deps'
                batches={[`ptc-3`]}
                conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }, `/claimedDependentsCount`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-1095-a-check-mfj-no-deps'
                batches={[`ptc-3`]}
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-1095-a-check-mfj-with-deps'
                batches={[`ptc-1`]}
                conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`]}
              />
              <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-1095-a-check-shared-2' />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <DFModal i18nKey='/info/credits-and-deductions/credits/is-1095-a-correct' />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/received-1095-a-without-marketplace-plan'
                batches={[`ptc-3`]}
              />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='multiple-forms-check'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-multiple-form-check' batches={[`ptc-1`]} />
              <DFAlert
                i18nKey={`/info/credits-and-deductions/credits/data-import-get-1095-a-single`}
                type={`info`}
                headingLevel={`h3`}
                conditions={[`/has1095AInIRDAS`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
              />
              <DFAlert
                i18nKey={`/info/credits-and-deductions/credits/data-import-get-1095-a-mfj`}
                type={`info`}
                headingLevel={`h3`}
                conditions={[`/has1095AInIRDAS`, `/isFilingStatusMFJ`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-multiple-form-check-shared'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/ptc-multiple-forms-self-no-deps'
                batches={[`ptc-1`]}
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/ptc-multiple-forms-self-with-deps'
                batches={[`ptc-1`]}
                conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }, `/claimedDependentsCount`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/ptc-multiple-forms-mfj-no-deps'
                batches={[`ptc-1`]}
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/ptc-multiple-forms-mfj-with-deps'
                batches={[`ptc-1`]}
                conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-with-state-plan'
                condition={`/isResidentOfStateWithMarketplace`}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/find-1095-a-without-state-plan'
                condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
              />
              <Boolean path='/writableSelfReportedHasMultiple1095As' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='multiple-forms-ko' condition='/selfReportedHasMultiple1095As' isKnockout>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-multiple-forms-ko' batches={[`ptc-1`]} />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen
              route='enrollment-family'
              condition={{
                operator: `isFalse`,
                condition: `/notMfjAndPrimaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
              }}
            >
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrollment-family-self-no-deps'
                batches={[`ptc-1`]}
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrollment-family-self-with-deps'
                batches={[`ptc-1`]}
                conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }, `/claimedDependentsCount`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrollment-family-mfj-no-deps'
                batches={[`ptc-1`]}
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrollment-family-mfj-with-deps'
                batches={[`ptc-1`]}
                conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`]}
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-enrollment-family' batches={[`ptc-1`]} />
              <Boolean path='/writableIsAdditionalPersonInEnrollmentFamily' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='enrollee-not-part-of-tax-family' condition='/isAdditionalPersonInEnrollmentFamily'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrollee-not-part-of-tax-family'
                batches={[`ptc-1`]}
              />
              <DFModal i18nKey='/info/credits-and-deductions/credits/what-is-tax-family' batches={[`ptc-3`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-enrollee-not-part-of-tax-family'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/what-if-enrolled-more-than-one'
                batches={[`ptc-1`]}
              />
              <Boolean path='/writableIsAdditionalPersonPartofDifferentTaxFamily' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='enrolled-with-other-family'
              condition={{ operator: `isFalse`, condition: `/isAdditionalPersonPartofDifferentTaxFamily` }}
            >
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrolled-with-other-family-self-no-deps'
                batches={[`ptc-1`]}
                conditions={[
                  { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrolled-with-other-family-self-with-deps'
                batches={[`ptc-1`]}
                conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }, `/claimedDependentsCount`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrolled-with-other-family-mfj-no-deps'
                batches={[`ptc-1`]}
                conditions={[
                  `/isFilingStatusMFJ`,
                  { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                ]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-enrolled-with-other-family-mfj-with-deps'
                batches={[`ptc-1`]}
                conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-enrolled-with-other-family'
                batches={[`ptc-3`]}
              />
              <Boolean path='/writableIsEnrolledWithOtherFamily' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='allocations-ko' condition='/needsAllocations' isKnockout>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/allocations-ko-tax-enrolled-with-other-family'
                batches={[`ptc-3`]}
                condition='/isEnrolledWithOtherFamily'
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/allocations-ko-enrollee-not-part-of-tax-family'
                batches={[`ptc-3`]}
                condition='/isAdditionalPersonPartofDifferentTaxFamily'
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/allocations-ko-shared' batches={[`ptc-1`]} />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen
              route='immigration-status-check'
              condition='/mayPotentiallyHaveTaxFamilyWithNonEligibleImmigrationStatus'
            >
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-immigration-status-check'
                batches={[`ptc-1`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/ptc-immigration-status-check'
                batches={[`ptc-1`]}
              />
              <DFModal
                i18nKey='/info/credits-and-deductions/credits/ptc-immigration-status-check'
                batches={[`ptc-1`]}
              />
              <Boolean path='/writableHasImmigrationStatusNotEligibleForMarketplace' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='immigration-status-ko' condition='/hasImmigrationStatusNotEligibleForMarketplace' isKnockout>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-immigration-ko' batches={[`ptc-1`]} />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen route='aptc-paid'>
              <MefAlert mefErrorCode='F8962-070' i18nKey='family-and-household' type='warning' />
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/ptc'
                batches={[`ptc-1`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-aptc-paid'
                batches={[`ptc-1`]}
                condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/credits/ptc-aptc-paid-mfj'
                batches={[`ptc-1`]}
                condition='/isFilingStatusMFJ'
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-advanced-payments-direct-insurance' />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-advanced-payments' batches={[`ptc-1`]} />
              <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-advanced-payments' batches={[`ptc-1`]} />
              <Boolean path='/writableHasAdvancedPtc' batches={[`ptc-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate
              condition={{ operator: `isFalse`, condition: `/filerNotQualifiedForPtcBecauseOfStatusAndResidenceInfo` }}
            >
              <Screen route='employer-plan-eligibility'>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-1`]}
                />
                <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-employer-plan' batches={[`ptc-1`]} />
                <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-employer-plan' batches={[`ptc-1`]} />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/ptc-what-is-employer-sponsored-coverage'
                  batches={[`ptc-3`]}
                />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/ptc-eligible-for-employer-sponsored-coverage'
                  batches={[`ptc-1`]}
                />
                <Boolean path='/writableHasEmployerSponsoredPlan' batches={[`ptc-1`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='employer-plan-ko' isKnockout condition='/hasEmployerSponsoredPlan'>
                <IconDisplay name='ErrorOutline' size={9} isCentered />
                <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-1`]} />
                <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-employer-plan-ko' batches={[`ptc-1`]} />
                <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
                <KnockoutButton i18nKey='button.knockout' />
              </Screen>
            </Gate>
          </Gate>
        </SubSubcategory>
        <Gate condition='/hasPtcQualifyingPlan'>
          <Gate
            condition={{ operator: `isFalse`, condition: `/filerNotQualifiedForPtcBecauseOfStatusAndResidenceInfo` }}
          >
            <SubSubcategory route='unique-circumstances'>
              <Screen route='mfs-exception-check' condition='/isMFSAndCantBeClaimed'>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-1`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-mfs-exception-check'
                  batches={[`ptc-1`]}
                />
                <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-mfs-exception-check' batches={[`ptc-3`]} />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/ptc-mfs-why-ask-abuse-abandonment'
                  batches={[`ptc-3`]}
                />
                <DFAlert
                  i18nKey='/info/credits-and-deductions/credits/ptc-mfs-concerns'
                  headingLevel='h2'
                  type='info'
                  batches={[`ptc-1`]}
                />
                <Boolean path='/writableHasMfsException' batches={[`ptc-1`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='mfs-exception-past-use' condition='/hasMfsException'>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-1`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-mfs-exception-past-use'
                  batches={[`ptc-1`]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-mfs-exception-past-use'
                  batches={[`ptc-1`]}
                />
                <Boolean path='/writableClaimingMfsExceptionForFourthStraightYear' batches={[`ptc-1`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='below-100-medicaid' condition='/maybeApplicableTpAndNoAptcAndBelow100PovertyThreshold'>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-1`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-below-100-fed-poverty-line-self-no-deps'
                  conditions={[
                    { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                    { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  ]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-below-100-fed-poverty-line-self-with-deps'
                  conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }, `/claimedDependentsCount`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-below-100-fed-poverty-line-mfj-no-deps'
                  conditions={[`/isFilingStatusMFJ`, { operator: `isFalse`, condition: `/claimedDependentsCount` }]} // uses falsy value of 0
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-below-100-fed-poverty-line-mfj-with-deps'
                  conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`]}
                />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/ptc-below-100-fed-poverty-line'
                  batches={[`ptc-3`]}
                />
                <Boolean
                  path='/writableBelow100ThresholdAndNotEligibleForMedicaidBecauseOfImmigrationStatus'
                  batches={[`ptc-1`]}
                />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='not-applicable-taxpayer-aptc' condition='/notApplicableTaxPayerAndHasAptc'>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-1`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-not-qualified-has-aptc'
                  batches={[`ptc-1`]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-not-qualified-has-aptc'
                  batches={[`ptc-1`]}
                />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
          </Gate>
          <Gate condition='/applicableTaxPayerOrHasAptc'>
            <SubSubcategory route='1095-a'>
              <Screen route='1095-collection-hub' hasScreenRouteOverride>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-2`]}
                />
                <TaxReturnAlert
                  type='error'
                  i18nKey='/info/credits-and-deductions/credits/ptc-need-1095-a'
                  batches={[`ptc-2`]}
                  conditions={[`/tpNeedsToAdd1095A`]}
                />
                <TaxReturnAlert
                  type='error'
                  i18nKey='/info/credits-and-deductions/credits/ptc-has-extra-1095-a'
                  batches={[`ptc-2`]}
                  conditions={[`/tpNeedsToDelete1095A`]}
                />
                <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-1095-intro' batches={[`ptc-2`]} />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-1095-self-no-deps'
                  batches={[`ptc-3`]}
                  conditions={[
                    { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                    { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  ]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-1095-self-with-deps'
                  batches={[`ptc-3`]}
                  conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }, `/claimedDependentsCount`]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-1095-mfj-no-deps'
                  batches={[`ptc-3`]}
                  conditions={[
                    `/isFilingStatusMFJ`,
                    { operator: `isFalse`, condition: `/claimedDependentsCount` }, // uses falsy value of 0
                  ]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-1095-mfj-with-deps'
                  batches={[`ptc-3`]}
                  conditions={[`/isFilingStatusMFJ`, `/claimedDependentsCount`]}
                />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/find-1095-a-with-state-plan'
                  condition={`/isResidentOfStateWithMarketplace`}
                />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/find-1095-a-without-state-plan'
                  condition={{ operator: `isFalse`, condition: `/isResidentOfStateWithMarketplace` }}
                />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/received-1095-a-without-marketplace-plan'
                  batches={[`ptc-3`]}
                />
                <CollectionItemManager path='/1095As' loopName='/1095As' donePath='/1095AsIsDone' />
              </Screen>
              <CollectionLoop
                loopName='/1095As'
                collection='/1095As'
                collectionItemCompletedCondition='/1095As/*/isComplete'
                shouldSeeHubCompletionBtnsPath='/hasAtLeastOneComplete1095A'
                donePath='/1095AsIsDone'
                iconName='Topic'
                isInner
                dataViewSections={[
                  {
                    i18nKey: `dataviews./flow/credits-and-deductions/credits/1095-a.has1095`,
                    condition: `/has1095A`,
                  },
                ]}
              >
                <SubSubcategory route='policy-details'>
                  <Screen route='policy-number'>
                    <ContextHeading
                      displayOnlyOn='edit'
                      i18nKey='/heading/credits-and-deductions/credits/ptc'
                      batches={[`ptc-2`]}
                    />
                    <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-policy-number' batches={[`ptc-2`]} />
                    <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-policy-number' batches={[`ptc-3`]} />
                    <LimitingString path='/1095As/*/policyNumber' batches={[`ptc-2`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen
                    route='marketplace-state'
                    condition={{ operator: `isFalseOrIncomplete`, condition: `/notPtcApplicableTaxPayer` }}
                  >
                    <ContextHeading
                      displayOnlyOn='edit'
                      i18nKey='/heading/credits-and-deductions/credits/ptc'
                      batches={[`ptc-2`]}
                    />
                    <Heading
                      i18nKey='/heading/credits-and-deductions/credits/ptc-marketplace-state'
                      batches={[`ptc-2`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/credits-and-deductions/credits/ptc-marketplace-state'
                      batches={[`ptc-2`]}
                    />
                    <Enum path='/1095As/*/marketplaceState' batches={[`ptc-2`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/notPtcApplicableTaxPayer` }}>
                    <Screen route='policy-dates'>
                      <ContextHeading
                        displayOnlyOn='edit'
                        i18nKey='/heading/credits-and-deductions/credits/ptc'
                        batches={[`ptc-2`]}
                      />
                      <TaxReturnAlert
                        type='error'
                        i18nKey='/info/credits-and-deductions/credits/ptc-fix-1095-a-dates'
                        condition='/1095As/*/hasPolicyDateError'
                      />
                      <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-policy-dates' batches={[`ptc-3`]} />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/credits/ptc-policy-dates'
                        batches={[`ptc-2`]}
                      />
                      <DatePicker path='/1095As/*/writablePolicyStartDate' lockYearTo='/taxYear' batches={[`ptc-2`]} />
                      <Boolean
                        path='/1095As/*/writableSpecialPolicyModificationInStartingMonth'
                        inputType='checkbox'
                        required={false}
                        batches={[`ptc-3`]}
                      />
                      <DatePicker path='/1095As/*/writablePolicyEndDate' lockYearTo='/taxYear' batches={[`ptc-2`]} />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Screen route='policy-dates-exception' condition='/1095As/*/maybeHasPolicyDateException'>
                      <ContextHeading
                        displayOnlyOn='edit'
                        i18nKey='/heading/credits-and-deductions/credits/ptc'
                        batches={[`ptc-2`]}
                      />
                      <Heading
                        i18nKey='/heading/credits-and-deductions/credits/ptc-policy-date-exception'
                        batches={[`ptc-2`]}
                      />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/credits/ptc-policy-date-exception'
                        batches={[`ptc-2`]}
                      />
                      <Boolean path='/1095As/*/writableHasPolicyDateException' batches={[`ptc-2`]} />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Screen
                      route='coverage-months-yn'
                      condition={{
                        operator: `isFalseOrIncomplete`,
                        condition: `/1095As/*/hasNoPotentialCoverageMonths`,
                      }}
                    >
                      <ContextHeading
                        displayOnlyOn='edit'
                        i18nKey='/heading/credits-and-deductions/credits/ptc'
                        batches={[`ptc-2`]}
                      />
                      <Heading
                        i18nKey='/heading/credits-and-deductions/credits/ptc-coverage-months'
                        batches={[`ptc-3`]}
                      />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months-pre'
                        batches={[`ptc-2`]}
                      />
                      <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months' batches={[`ptc-2`]} />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months'
                        batches={[`ptc-2`]}
                      />
                      <Boolean path='/1095As/*/hasCoverageMonths' batches={[`ptc-2`]} />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Screen route='zero-coverage-months' condition='/1095As/*/hasNoCoverageMonthsAndNoAptc'>
                      <ContextHeading
                        displayOnlyOn='edit'
                        i18nKey='/heading/credits-and-deductions/credits/ptc'
                        batches={[`ptc-2`]}
                      />
                      <Heading
                        i18nKey='/heading/credits-and-deductions/credits/ptc-zero-coverage-months'
                        batches={[`ptc-2`]}
                      />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/credits/ptc-zero-coverage-months'
                        batches={[`ptc-2`]}
                      />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Gate condition='/1095As/*/hasCoverageMonthAndHasPotentialCoverageMonths'>
                      <Screen route='enrollee-coverage-months'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-enrollee-coverage-months'
                          batches={[`ptc-2`]}
                        />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months'
                          batches={[`ptc-2`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months'
                          batches={[`ptc-2`]}
                        />
                        <MultiEnum path='/1095As/*/coverageMonths' batches={[`ptc-2`]} displayOnlyOn='edit' />
                        <GenericString
                          path='/1095As/*/numberOfCoverageMonths'
                          batches={[`ptc-2`]}
                          displayOnlyOn='data-view'
                        />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='coverage-months'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-coverage-months-qualified'
                          batches={[`ptc-2`]}
                        />
                        <ConditionalList
                          i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months-qualified'
                          items={[
                            { itemKey: `jan`, conditions: [`/1095As/*/hasJanCoverageMonth`] },
                            { itemKey: `feb`, conditions: [`/1095As/*/hasFebCoverageMonth`] },
                            { itemKey: `mar`, conditions: [`/1095As/*/hasMarCoverageMonth`] },
                            { itemKey: `apr`, conditions: [`/1095As/*/hasAprCoverageMonth`] },
                            { itemKey: `may`, conditions: [`/1095As/*/hasMayCoverageMonth`] },
                            { itemKey: `jun`, conditions: [`/1095As/*/hasJunCoverageMonth`] },
                            { itemKey: `jul`, conditions: [`/1095As/*/hasJulCoverageMonth`] },
                            { itemKey: `aug`, conditions: [`/1095As/*/hasAugCoverageMonth`] },
                            { itemKey: `sep`, conditions: [`/1095As/*/hasSepCoverageMonth`] },
                            { itemKey: `oct`, conditions: [`/1095As/*/hasOctCoverageMonth`] },
                            { itemKey: `nov`, conditions: [`/1095As/*/hasNovCoverageMonth`] },
                            { itemKey: `dec`, conditions: [`/1095As/*/hasDecCoverageMonth`] },
                          ]}
                          batches={[`ptc-2`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-coverage-months-qualified'
                          batches={[`ptc-2`]}
                        />
                        <SaveAndOrContinueButton />
                      </Screen>
                    </Gate>
                  </Gate>
                </SubSubcategory>
                <SubSubcategory route='premiums-and-aptc'>
                  <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/notPtcApplicableTaxPayer` }}>
                    <Gate condition='/1095As/*/hasAtLeastOneCoverageMonth'>
                      <Screen route='annual-monthly' condition='/1095As/*/hasPolicyForAllMonths'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-did-premiums-change'
                          batches={[`ptc-2`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-did-premiums-change'
                          batches={[`ptc-2`]}
                        />
                        <Boolean path='/1095As/*/writablePremiumsOrSlcspChange' batches={[`ptc-2`]} />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen
                        route='annual-premium'
                        condition={{ operator: `isFalse`, condition: `/1095As/*/premiumsOrSlcspChange` }}
                      >
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-annual-premiums'
                          batches={[`ptc-2`]}
                        />
                        <Dollar path='/1095As/*/writablePtcAnnualPremium' batches={[`ptc-2`]} />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen
                        route='monthly-premium'
                        condition={{
                          operator: `isTrueOrIncomplete`,
                          condition: `/1095As/*/premiumsOrSlcspChange`,
                        }}
                      >
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-monthly-premiums'
                          batches={[`ptc-3`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-monthly-premiums'
                          batches={[`ptc-3`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumJan'
                          condition='/1095As/*/policyIsActiveInJan'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumFeb'
                          condition='/1095As/*/policyIsActiveInFeb'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumMar'
                          condition='/1095As/*/policyIsActiveInMar'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumApr'
                          condition='/1095As/*/policyIsActiveInApr'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumMay'
                          condition='/1095As/*/policyIsActiveInMay'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumJun'
                          condition='/1095As/*/policyIsActiveInJun'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumJul'
                          condition='/1095As/*/policyIsActiveInJul'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumAug'
                          condition='/1095As/*/policyIsActiveInAug'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumSep'
                          condition='/1095As/*/policyIsActiveInSep'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumOct'
                          condition='/1095As/*/policyIsActiveInOct'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumNov'
                          condition='/1095As/*/policyIsActiveInNov'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyPremiumDec'
                          condition='/1095As/*/policyIsActiveInDec'
                          batches={[`ptc-2`]}
                        />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='change-circumstances'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-change-circumstances'
                          batches={[`ptc-3`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-change-circumstances'
                          batches={[`ptc-3`]}
                        />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-change-circumstances'
                          batches={[`ptc-3`]}
                        />
                        <Boolean path='/1095As/*/hasChangeInCircumstance' batches={[`ptc-2`]} />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='annual-slcsp' condition='/1095As/*/noChangesAnd12MonthPolicy'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-annual-slcsp'
                          batches={[`ptc-2`]}
                        />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-what-if-slcsp-inaccurate-state'
                          batches={[`ptc-2`]}
                          condition='/1095As/*/hasStateMarketplacePlan'
                        />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-what-if-slcsp-inaccurate-federal'
                          batches={[`ptc-2`]}
                          condition={{ operator: `isFalse`, condition: `/1095As/*/hasStateMarketplacePlan` }}
                        />
                        <Dollar path='/1095As/*/writableSlcspAnnualPremium' batches={[`ptc-2`]} />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='look-up-slcsp' condition='/1095As/*/hasChangeInCircumstance'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-lookup-slcsp'
                          batches={[`ptc-2`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-lookup-slcsp-state'
                          batches={[`ptc-2`]}
                          condition='/1095As/*/hasStateMarketplacePlan'
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-lookup-slcsp-federal'
                          batches={[`ptc-2`]}
                          condition={{ operator: `isFalse`, condition: `/1095As/*/hasStateMarketplacePlan` }}
                        />
                        <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-what-is-slcsp' batches={[`ptc-2`]} />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-how-to-check-state-slcsp'
                          batches={[`ptc-2`]}
                          condition='/1095As/*/hasStateMarketplacePlan'
                        />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='monthly-slcsp' condition='/1095As/*/hasChangeOrPolicyIsNotForAllMonths'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-monthly-slcsp'
                          batches={[`ptc-2`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-slcsp-change-circumstance-state'
                          batches={[`ptc-2`]}
                          conditions={[`/1095As/*/hasChangeInCircumstance`, `/1095As/*/hasStateMarketplacePlan`]}
                        />
                        <InfoDisplay
                          i18nKey='/info/credits-and-deductions/credits/ptc-slcsp-change-circumstance-federal'
                          batches={[`ptc-2`]}
                          conditions={[
                            `/1095As/*/hasChangeInCircumstance`,
                            { operator: `isFalse`, condition: `/1095As/*/hasStateMarketplacePlan` },
                          ]}
                        />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-what-if-slcsp-inaccurate-state'
                          batches={[`ptc-2`]}
                          conditions={[
                            { operator: `isFalse`, condition: `/1095As/*/hasChangeInCircumstance` },
                            `/1095As/*/hasStateMarketplacePlan`,
                          ]}
                        />
                        <DFModal
                          i18nKey='/info/credits-and-deductions/credits/ptc-what-if-slcsp-inaccurate-federal'
                          batches={[`ptc-2`]}
                          conditions={[
                            { operator: `isFalse`, condition: `/1095As/*/hasChangeInCircumstance` },
                            { operator: `isFalse`, condition: `/1095As/*/hasStateMarketplacePlan` },
                          ]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumJan'
                          conditions={[`/1095As/*/policyIsActiveInJan`, `/1095As/*/hasJanCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderJan'
                          conditions={[
                            `/1095As/*/policyIsActiveInJan`,
                            { operator: `isFalse`, condition: `/1095As/*/hasJanCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumFeb'
                          conditions={[`/1095As/*/policyIsActiveInFeb`, `/1095As/*/hasFebCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderFeb'
                          conditions={[
                            `/1095As/*/policyIsActiveInFeb`,
                            { operator: `isFalse`, condition: `/1095As/*/hasFebCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumMar'
                          conditions={[`/1095As/*/policyIsActiveInMar`, `/1095As/*/hasMarCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderMar'
                          conditions={[
                            `/1095As/*/policyIsActiveInMar`,
                            { operator: `isFalse`, condition: `/1095As/*/hasMarCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumApr'
                          conditions={[`/1095As/*/policyIsActiveInApr`, `/1095As/*/hasAprCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderApr'
                          conditions={[
                            `/1095As/*/policyIsActiveInApr`,
                            { operator: `isFalse`, condition: `/1095As/*/hasAprCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumMay'
                          conditions={[`/1095As/*/policyIsActiveInMay`, `/1095As/*/hasMayCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderMay'
                          conditions={[
                            `/1095As/*/policyIsActiveInMay`,
                            { operator: `isFalse`, condition: `/1095As/*/hasMayCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumJun'
                          conditions={[`/1095As/*/policyIsActiveInJun`, `/1095As/*/hasJunCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderJun'
                          conditions={[
                            `/1095As/*/policyIsActiveInJun`,
                            { operator: `isFalse`, condition: `/1095As/*/hasJunCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumJul'
                          conditions={[`/1095As/*/policyIsActiveInJul`, `/1095As/*/hasJulCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderJul'
                          conditions={[
                            `/1095As/*/policyIsActiveInJul`,
                            { operator: `isFalse`, condition: `/1095As/*/hasJulCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumAug'
                          conditions={[`/1095As/*/policyIsActiveInAug`, `/1095As/*/hasAugCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderAug'
                          conditions={[
                            `/1095As/*/policyIsActiveInAug`,
                            { operator: `isFalse`, condition: `/1095As/*/hasAugCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumSep'
                          conditions={[`/1095As/*/policyIsActiveInSep`, `/1095As/*/hasSepCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderSep'
                          conditions={[
                            `/1095As/*/policyIsActiveInSep`,
                            { operator: `isFalse`, condition: `/1095As/*/hasSepCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumOct'
                          conditions={[`/1095As/*/policyIsActiveInOct`, `/1095As/*/hasOctCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderOct'
                          conditions={[
                            `/1095As/*/policyIsActiveInOct`,
                            { operator: `isFalse`, condition: `/1095As/*/hasOctCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumNov'
                          conditions={[`/1095As/*/policyIsActiveInNov`, `/1095As/*/hasNovCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderNov'
                          conditions={[
                            `/1095As/*/policyIsActiveInNov`,
                            { operator: `isFalse`, condition: `/1095As/*/hasNovCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlySlcspPremiumDec'
                          conditions={[`/1095As/*/policyIsActiveInDec`, `/1095As/*/hasDecCoverageMonth`]}
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/slcspPremiumPlaceholderDec'
                          conditions={[
                            `/1095As/*/policyIsActiveInDec`,
                            { operator: `isFalse`, condition: `/1095As/*/hasDecCoverageMonth` },
                          ]}
                          hintKey='/info/credits-and-deductions/credits/why-cant-i-change-slcsp-amount'
                          batches={[`ptc-3`]}
                          readOnly
                        />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='monthly-aptc' condition='/1095As/*/showMonthlyAptc'>
                        <ContextHeading
                          displayOnlyOn='edit'
                          i18nKey='/heading/credits-and-deductions/credits/ptc'
                          batches={[`ptc-2`]}
                        />
                        <Heading
                          i18nKey='/heading/credits-and-deductions/credits/ptc-monthly-aptc'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumJan'
                          condition='/1095As/*/policyIsActiveInJan'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumFeb'
                          condition='/1095As/*/policyIsActiveInFeb'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumMar'
                          condition='/1095As/*/policyIsActiveInMar'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumApr'
                          condition='/1095As/*/policyIsActiveInApr'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumMay'
                          condition='/1095As/*/policyIsActiveInMay'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumJun'
                          condition='/1095As/*/policyIsActiveInJun'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumJul'
                          condition='/1095As/*/policyIsActiveInJul'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumAug'
                          condition='/1095As/*/policyIsActiveInAug'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumSep'
                          condition='/1095As/*/policyIsActiveInSep'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumOct'
                          condition='/1095As/*/policyIsActiveInOct'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumNov'
                          condition='/1095As/*/policyIsActiveInNov'
                          batches={[`ptc-2`]}
                        />
                        <Dollar
                          path='/1095As/*/writablePtcMonthlyAptcPremiumDec'
                          condition='/1095As/*/policyIsActiveInDec'
                          batches={[`ptc-2`]}
                        />
                        <SaveAndOrContinueButton />
                      </Screen>
                    </Gate>
                  </Gate>
                  <Screen route='annual-aptc' condition='/1095As/*/hasAptcAndPolicyIsFullYearOrZeroCoverageMonths'>
                    <ContextHeading
                      displayOnlyOn='edit'
                      i18nKey='/heading/credits-and-deductions/credits/ptc'
                      batches={[`ptc-2`]}
                    />
                    <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-annual-aptc' batches={[`ptc-2`]} />
                    <Dollar path='/1095As/*/writableAptcAnnualAmount' batches={[`ptc-2`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='collected-1095' condition='/1095As/*/hasCoverageMonthsOrAptc'>
                    <ContextHeading
                      displayOnlyOn='edit'
                      i18nKey='/heading/credits-and-deductions/credits/ptc'
                      batches={[`ptc-2`]}
                    />
                    <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-end-of-1095-a' batches={[`ptc-2`]} />
                    <SetFactAction path='/1095As/*/hasSeenLastAvailableScreen' source='/flowTrue' />
                    <SaveAndOrContinueButton />
                  </Screen>
                </SubSubcategory>
              </CollectionLoop>
            </SubSubcategory>
            <SubSubcategory route='alt-calc-marriage'>
              <Screen route='alt-calc-mfj-excess-aptc' condition='/potentiallyEligibleForAltCalcBecauseMarriedThisYear'>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/ptc'
                  batches={[`ptc-2`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/credits/ptc-alt-calc-excess-aptc'
                  batches={[`ptc-2`]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/ptc-alt-calc-excess-aptc'
                  batches={[`ptc-2`]}
                />
                <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-alt-calc-excess-aptc' batches={[`ptc-2`]} />
                <Boolean path='/writableWantsToUseAltCalculations' batches={[`ptc-2`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='alt-calc-ko' condition='/wantsToUseAltCalculations' isKnockout>
                <IconDisplay name='ErrorOutline' size={9} isCentered />
                <Heading i18nKey='/heading/knockout/ptc-generic-ko' batches={[`ptc-2`]} />
                <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-alt-calc-ko' batches={[`ptc-2`]} />
                <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
                <KnockoutButton i18nKey='button.knockout' />
              </Screen>
            </SubSubcategory>
          </Gate>
        </Gate>
        <SubSubcategory route='ptc-outcomes-not-qualified' editable={false} headingLevel='h3'>
          <Assertion
            type='info'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-not-qualified-no-aptc'
            conditions={[
              `/eitherDependentTpForm8962NotNeededOrNotQualifiedNoAptc`,
              `/maybeEligibleForPtc`,
              `/ptcSectionIsComplete`,
            ]}
          />
          <Assertion
            type='info'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-not-qualified-aptc'
            conditions={[
              `/ptcNotQualifiedHasAptc`,
              { operator: `isFalseOrIncomplete`, condition: `/isRepaymentLimitationIsActive` },
              `/maybeEligibleForPtc`,
              `/ptcSectionIsComplete`,
            ]}
          />
          <Assertion
            type='info'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-not-qualified-aptc-repayment-limitation'
            conditions={[
              `/ptcNotQualifiedHasAptc`,
              `/isRepaymentLimitationIsActive`,
              `/maybeEligibleForPtc`,
              `/ptcSectionIsComplete`,
            ]}
          />
          <Screen route='dep-tp-8962-not-needed' condition='/dependentTpForm8962NotNeeded'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-dep-tp-8962-not-needed' batches={[`ptc-1`]} />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-did-not-enroll-self-or-others'
              batches={[`ptc-1`]}
              condition='/ptcDependentTpDidNotEnrollSelfOrOthers'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-did-is-claimed-enrolled-self-not-others'
              batches={[`ptc-1`]}
              condition='/ptcDependentTpIsClaimedDidEnrollSelfAndNotOthers'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-is-claimed-did-enroll-self-and-others-and-someone-part-of-diff-tax-family'
              batches={[`ptc-1`]}
              condition='/ptcDependentTpIsClaimedDidEnrollSelfAndOthersAndPartOfDifferentTaxFamily'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-dep-tp-did-not-enroll-self-enrolled-others-and-part-of-diff-tax-family'
              batches={[`ptc-1`]}
              condition='/ptcDependentTpDidNotEnrollSelfAndEnrolledOthersWhoIsPartOfDifferentTaxFamily'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='ptc-not-qualified-no-aptc' condition='/ptcNotQualifiedNoAptc'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-not-qualified' batches={[`ptc-1`]} />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/why-i-dont-qualify' batches={[`ptc-1`]} />
            <ConditionalList
              i18nKey='/info/credits-and-deductions/credits/intro-no-credits/reasons'
              items={PtcDisqualifyingItems}
              batches={[`ptc-1`]}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='ptc-not-qualified-aptc' condition='/ptcNotQualifiedHasAptc'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-2`]}
            />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-not-qualified-aptc' batches={[`ptc-3`]} />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/why-i-dont-qualify' batches={[`ptc-2`]} />
            <ConditionalList
              i18nKey='/info/credits-and-deductions/credits/intro-no-credits/reasons'
              items={PtcDisqualifyingItems}
              batches={[`ptc-2`]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-how-we-calculated-summary'
              condition='/isRepaymentLimitationIsActive'
              batches={[`ptc-2`]}
            />
            <SummaryTable
              i18nKey='/info/credits-and-deductions/credits/ptc-repayment-limitation'
              batches={[`ptc-2`]}
              condition='/isRepaymentLimitationIsActive'
              items={[{ itemKey: `ptc` }, { itemKey: `aptc` }, { itemKey: `total`, showTopBorder: true }]}
            />
            <DFModal i18nKey='/info/credits-and-deductions/credits/ptc-excess-aptc' batches={[`ptc-2`]} />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-excess-aptc' batches={[`ptc-2`]} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='ptc-outcomes-qualified' editable={false} headingLevel='h3'>
          <Assertion
            type='success'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-qualified-no-aptc'
            conditions={[`/qualifiedForPtcNoAptcAndPtcTotalGreaterThanZero`, `/ptcSectionIsComplete`]}
          />
          <Assertion
            type='success'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-qualified-no-aptc-zero-credit'
            conditions={[`/qualifiedForPtcNoAptcAndPtcTotalIsZero`, `/ptcSectionIsComplete`]}
          />
          <Assertion
            type='success'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-qualified-credit-equals-aptc'
            conditions={[`/qualifiedForPtcHasAptcAndNetCreditIsZero`, `/ptcSectionIsComplete`]}
          />
          <Assertion
            type='success'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-qualified-positive-net-ptc'
            conditions={[`/qualifiedForPtcHasAptcAndNetCreditIsPositive`, `/ptcSectionIsComplete`]}
          />
          <Assertion
            type='success'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-qualified-excess-aptc'
            conditions={[
              `/qualifiedForPtcHasAptcAndNetCreditIsNegativeAndRepaymentLimitationIsNotActive`,
              `/ptcSectionIsComplete`,
            ]}
          />
          <Assertion
            type='success'
            i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ptc-qualified-excess-aptc-repayment-limitation'
            conditions={[
              `/qualifiedForPtcHasAptcAndNetCreditIsNegativeAndHasRepaymentLimitation`,
              `/ptcSectionIsComplete`,
            ]}
          />

          <Screen route='ptc-no-aptc' condition='/qualifiedForPtcNoAptcAndPtcTotalGreaterThanZero'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-no-aptc' batches={[`ptc-1`]} />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-no-aptc-total-ptc-greater-than-zero'
              batches={[`ptc-1`]}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='ptc-no-aptc-zero-credit' condition='/qualifiedForPtcNoAptcAndPtcTotalIsZero'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-no-aptc' batches={[`ptc-1`]} />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-no-aptc-total-ptc-is-zero'
              batches={[`ptc-1`]}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='ptc-equals-aptc' condition='/qualifiedForPtcHasAptcAndNetCreditIsZero'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-aptc-zero-net-credits'
              batches={[`ptc-3`]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-aptc-zero-net-credits'
              batches={[`ptc-1`]}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='net-ptc' condition='/qualifiedForPtcHasAptcAndNetCreditIsPositive'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-aptc-positive-net-credits'
              batches={[`ptc-1`]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-how-we-calculated-summary'
              batches={[`ptc-1`]}
            />
            <SummaryTable
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-aptc-positive-net-credits'
              batches={[`ptc-1`]}
              items={[{ itemKey: `ptc` }, { itemKey: `aptc` }, { itemKey: `total`, showTopBorder: true }]}
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-aptc-positive-net-credits'
              batches={[`ptc-1`]}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='excess-aptc'
            condition='/qualifiedForPtcHasAptcAndNetCreditIsNegativeAndRepaymentLimitationIsNotActive'
          >
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/ptc'
              batches={[`ptc-1`]}
            />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-excess-aptc' batches={[`ptc-3`]} />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-how-we-calculated-summary'
              batches={[`ptc-1`]}
            />
            <SummaryTable
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-excess-aptc-no-limitation'
              batches={[`ptc-1`]}
              items={[{ itemKey: `ptc` }, { itemKey: `aptc` }, { itemKey: `total`, showTopBorder: true }]}
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-excess-aptc-no-link' batches={[`ptc-1`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='excess-aptc-repay-limitation'
            condition='/qualifiedForPtcHasAptcAndNetCreditIsNegativeAndHasRepaymentLimitation'
          >
            <ContextHeading i18nKey='/heading/credits-and-deductions/credits/ptc' batches={[`ptc-1`]} />
            <Heading i18nKey='/heading/credits-and-deductions/credits/ptc-qualified-excess-aptc' batches={[`ptc-3`]} />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/ptc-how-we-calculated-summary'
              batches={[`ptc-1`]}
            />
            <SummaryTable
              i18nKey='/info/credits-and-deductions/credits/ptc-qualified-excess-aptc-with-limitation'
              batches={[`ptc-1`]}
              items={[
                { itemKey: `ptc` },
                { itemKey: `aptc` },
                { itemKey: `totalExcessAptc`, showTopBorder: true },
                { itemKey: `totalNetCredit` },
              ]}
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/ptc-excess-aptc-no-link' batches={[`ptc-1`]} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Gate>

      {/* Child and Dependent Care Credit (CDCC) */}
      {CdccSubSubcategory}

      {/* Elderly or Disabled Credit (EDC) */}
      {EdcSubSubcategory}

      {/* Saver's Credit  */}
      {SaversCreditSubSubcategory}

      <Screen route='nr-credit-limit-reached' condition='/flowCouldHaveQualifiedForAdditionalNrCredit'>
        <Heading i18nKey='/heading/credits-and-deductions/credits/nr-credit-limit-reached' batches={[`credits-ia-1`]} />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/nr-credit-limit-reached'
          batches={[`credits-ia-1`]}
        />
        <ConditionalList
          i18nKey='/info/credits-and-deductions/credits/nr-credit-limit-reached'
          items={[
            {
              itemKey: `cdcc`,
              conditions: [`/isReceivingCdccCredit`],
            },
            {
              itemKey: `edc`,
              conditions: [`/qualifiedForCreditForElderlyAndDisabled`],
            },
            {
              itemKey: `savers`,
              conditions: [`/qualifiedForSaverCredit`],
            },
          ]}
          batches={[`credits-ia-1`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/nr-credit-limit-reached-conclusion'
          batches={[`credits-ia-1`]}
        />
        <SaveAndOrContinueButton />
      </Screen>

      {/* CTC and ODC */}
      {CtcOdcSubSubcategory}

      {/* Earned Income Tax Credit (EITC) */}
      {EitcSubSubcategory}

      {/* Credit Summary */}
      <SubSubcategory route='summary-of-credits' editable={false} borderStyle='heavy'>
        <Gate condition='/creditsSectionComplete'>
          <Screen route='credit-summary-qualified' condition='/hasQualifiedForAtLeastOneCredit'>
            <Heading
              batches={[`credits-ia-1`]}
              i18nKey='/heading/credits-and-deductions/credits/credit-summary-qualified'
            />
            <SummaryTable
              i18nKey='/info/credits-and-deductions/credits/summary-nonrefundable'
              batches={[`information-architecture-0`]}
              condition='/qualifiedForNonrefundableCredit'
              items={CreditsSummaryNonRefundableItems}
            />
            <SummaryTable
              i18nKey='/info/credits-and-deductions/credits/summary-refundable'
              condition='/qualifiedForRefundableCredit'
              items={CreditsSummaryRefundableItems}
              batches={[`information-architecture-0`]}
            />
            {CreditsDisqualificationAccordion}
            {/* Credit Summary */}
            <Dollar
              path='/netPtcAmountWhenZeroOrPositive'
              conditions={[`/needsToFileForm8962`, `/netPtcAmountIsZeroOrPositive`]}
              batches={[`ptc-1`]}
              displayOnlyOn='data-view'
            />
            <Dollar
              path='/ptcAmountOwedDisplayedAsZero'
              conditions={[`/needsToFileForm8962`, `/netPtcAmountIsNegative`]}
              batches={[`ptc-1`]}
              displayOnlyOn='data-view'
            />
            <Dollar path='/cdccTotalCredit' conditions={[`/cdccQualified`]} displayOnlyOn='data-view' />
            <Dollar path='/totalEdc' condition='/qualifiedForCreditForElderlyAndDisabled' displayOnlyOn='data-view' />
            <Dollar
              path='/qualifiedSaverCreditAmount'
              batches={[`savers-8880-0`]}
              condition='/qualifiedForSaverCredit'
              displayOnlyOn='data-view'
            />
            <Dollar path='/totalOdc' condition='/odcQualified' displayOnlyOn='data-view' />
            <Dollar path='/totalCtc' condition='/ctcQualified' displayOnlyOn='data-view' />
            <Dollar path='/additionalCtc' condition='/ctcQualified' displayOnlyOn='data-view' />
            <Dollar path='/earnedIncomeCredit' condition='/eitcQualified' displayOnlyOn='data-view' />
            <Dollar
              path='/ptcAmountOwedDisplayedAsZero'
              condition={{ operator: `isFalse`, condition: `/needsToFileForm8962` }}
              displayOnlyOn='data-view'
              batches={[`ptc-1`]}
              hasZeroOverride={true}
            />
            <Dollar
              path='/cdccTotalCredit'
              condition={{ operator: `isFalse`, condition: `/cdccQualified` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <Dollar
              path='/totalEdc'
              condition={{ operator: `isFalse`, condition: `/qualifiedForCreditForElderlyAndDisabled` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <Dollar
              path='/qualifiedSaverCreditAmount'
              batches={[`savers-8880-0`]}
              condition={{ operator: `isFalse`, condition: `/qualifiedForSaverCredit` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <Dollar
              path='/totalOdc'
              condition={{ operator: `isFalse`, condition: `/odcQualified` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <Dollar
              path='/totalCtc'
              condition={{ operator: `isFalse`, condition: `/ctcQualified` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <Dollar
              path='/additionalCtc'
              condition={{ operator: `isFalse`, condition: `/ctcQualified` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <Dollar
              path='/earnedIncomeCredit'
              condition={{ operator: `isFalse`, condition: `/eitcQualified` }}
              displayOnlyOn='data-view'
              hasZeroOverride={true}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='credit-summary-not-qualified'
            condition={{ operator: `isFalse`, condition: `/hasQualifiedForAtLeastOneCredit` }}
          >
            <Heading i18nKey='/heading/credits-and-deductions/credits/intro-no-credits' batches={[`credits-ia-1`]} />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/intro-no-credits'
              batches={[`credits-ia-1`, `cdcc-0`]}
            />
            {CreditsDisqualificationAccordion}
            <DFAccordion
              i18nKey='/info/credits-and-deductions/credits/intro-no-credits/dont-qualify-mfj-dep'
              condition='/isMFJDependent'
              batches={[`cdcc-2`]}
              internalLink='/flow/you-and-your-family/spouse/spouse-mfj-basic-info'
            />
            <InfoDisplay
              i18nKey='/info/credits-and-deductions/credits/intro-no-credits'
              batches={[`savers-8880-0`, `credits-ia-1`, `cdcc-0`]}
            />
            <HelpLink i18nKey='/info/credits-and-deductions/credits/intro-no-credits' />
            <Dollar path='/ptcAmountOwedDisplayedAsZero' displayOnlyOn='data-view' hasZeroOverride={true} />
            <Dollar path='/cdccTotalCredit' hasZeroOverride={true} batches={[`cdcc-0`]} displayOnlyOn='data-view' />
            <Dollar path='/totalEdc' hasZeroOverride={true} displayOnlyOn='data-view' batches={[`credits-ia-1`]} />
            <Dollar
              path='/qualifiedSaverCreditAmount'
              hasZeroOverride={true}
              batches={[`savers-8880-0`]}
              displayOnlyOn='data-view'
            />
            <Dollar path='/totalOdc' hasZeroOverride={true} displayOnlyOn='data-view' />
            <Dollar path='/totalCtc' hasZeroOverride={true} displayOnlyOn='data-view' />
            <Dollar path='/additionalCtc' hasZeroOverride={true} displayOnlyOn='data-view' />
            <Dollar path='/earnedIncomeCredit' hasZeroOverride={true} displayOnlyOn='data-view' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </SubSubcategory>
    </Gate>
    <Gate condition={{ operator: `isFalse`, condition: `/hasPotentialCredits` }}>
      <SubSubcategory route='summary-of-credits'>
        <Screen route='credits-intro-no-credits' actAsDataView={true}>
          <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
          <ContextHeading i18nKey='/heading/credits-and-deductions/credits/credits' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/intro-no-credits' batches={[`cdcc-2`]} />
          <DFModal
            i18nKey='/info/credits-and-deductions/credits/intro-no-credits'
            batches={[`savers-8880-0`, `credits-ia-1`, `cdcc-2`, `ptc-2`]}
          />
          {CreditsDisqualificationAccordion}
          <DFAccordion
            i18nKey='/info/credits-and-deductions/credits/intro-no-credits/dont-qualify-mfj-dep'
            condition='/isMFJDependent'
            batches={[`cdcc-2`]}
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/intro-no-credits' batches={[`cdcc-2`]} />
          <HelpLink i18nKey='/info/credits-and-deductions/credits/intro-no-credits' batches={[`cdcc-2`]} />
          <SetFactAction path='/flowHasSeenCreditsIntroNoCredits' source='/flowTrue' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </Gate>
    <Screen route='remove-family-hh' condition='/hasAtLeastOneDependentWhoDoesntQualifyForAnyBenefits'>
      <ContextHeading i18nKey='/heading/credits-and-deductions/credits/credits' />
      <Heading i18nKey='/heading/credits-and-deductions/credits/family-and-household' />
      <InfoDisplay i18nKey='/info/credits-and-deductions/credits/family-and-household' />
      <ConditionalList
        i18nKey='/info/credits-and-deductions/credits/non-qualifying-dependents'
        items={[{ itemKey: `primary`, collection: `/familyAndHouseholdWhoDontQualifyForAnyBenefitsCollection` }]}
      />
      <InternalLink
        i18nKey='/info/credits-and-deductions/credits/go-back-to-family-and-household'
        route='/data-view/flow/you-and-your-family/dependents'
      />
      <SaveAndOrContinueButton />
    </Screen>
  </Subcategory>
);
