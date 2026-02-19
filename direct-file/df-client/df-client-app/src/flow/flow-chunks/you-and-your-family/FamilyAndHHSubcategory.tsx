/* eslint-disable max-len */
import { CollectionLoop, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  FactAssertion,
  Boolean,
  CollectionItemManager,
  ConditionalAccordion,
  ConditionalList,
  ContextHeading,
  DatePicker,
  DFAlert,
  DFModal,
  Enum,
  FactResultAssertion,
  GenericString,
  Heading,
  HelpLink,
  IconDisplay,
  InfoDisplay,
  InternalLink,
  IntroContent,
  LimitingString,
  IpPin,
  SaveAndOrContinueButton,
  SetFactAction,
  TaxReturnAlert,
  Tin,
  MefAlert,
  KnockoutButton,
  DFAccordion,
} from '../../ContentDeclarations.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';

const DependentDoesntQualifyAccordion = (
  <ConditionalAccordion
    i18nKey='/info/you-and-your-family/dependents/reasons'
    displayOnlyHeaders
    items={[
      {
        itemKey: `subListNotQcOrRelative`,
        conditions: [{ operator: `isFalse`, condition: `/familyAndHousehold/*/qcOrQr` }],
      },
      {
        itemKey: `subListDependentTaxpayerTest`,
        conditions: [`/treatFilersAsDependents`],
      },
      {
        itemKey: `subListCustodialParent`,
        conditions: [
          `/familyAndHousehold/*/tpCanClaimSplitBenefitsHohEitcCdcc`,
          `/familyAndHousehold/*/tpIsCustodialParent`,
        ],
      },
      {
        itemKey: `subListOtherEligibleTaxpayer`,
        conditions: [
          `/familyAndHousehold/*/tpCanClaimSplitBenefitsHohEitcCdcc`,
          `/familyAndHousehold/*/reachedNonParentalBenefitSplit`,
        ],
      },
      {
        itemKey: `subListNoTin`,
        conditions: [`/familyAndHousehold/*/hasTinTypeOfNone`],
      },
      {
        itemKey: `subListCitizenshipOrResidency`,
        conditions: [{ operator: `isFalse`, condition: `/familyAndHousehold/*/citizenOrResidentTest` }],
      },
      {
        itemKey: `subListJointReturnTest`,
        conditions: [
          { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/qcOrQr` },
          { operator: `isFalse`, condition: `/familyAndHousehold/*/jointReturnTest` },
        ],
      },
      {
        itemKey: `subListDisqualifiedDueToParentalCustody`,
        conditions: [
          `/familyAndHousehold/*/disqualifiedDueToParentalCustody`,
          { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/specialRuleAgiTest` },
        ],
      },
      {
        itemKey: `subListDisqualifiedDueToParentalCustodyAGI`,
        conditions: [
          `/familyAndHousehold/*/disqualifiedDueToParentalCustody`,
          { operator: `isFalse`, condition: `/familyAndHousehold/*/specialRuleAgiTest` },
        ],
      },
    ]}
  />
);

// These are used in the "you may qualify" conditional list on qualified-dependent and potential-qp
// This is a list of all credits this person may qualify the tp for in the future.
// (We do not know income or filing status or claim status at this point)
const mayQualifyTPForTaxBenefitsItems: ItemConfig[] = [
  {
    itemKey: `hoh`,
    conditions: [{ operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/personQualifiesTPforHoH` }],
  },
  {
    itemKey: `qss`,
    conditions: [{ operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/personQualifiesTPforQss` }],
  },
  {
    itemKey: `cdcc`,
    conditions: [
      {
        operator: `isTrueOrIncomplete`,
        condition: `/familyAndHousehold/*/personCouldQualifyTPForCdcc`,
      },
    ],
  },
  {
    itemKey: `ctc`,
    conditions: [{ operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/eligibleCtc` }],
  },
  {
    itemKey: `odc`,
    conditions: [{ operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/eligibleOdc` }],
  },
  {
    itemKey: `eitc`,
    conditions: [
      {
        operator: `isTrueOrIncomplete`,
        condition: `/familyAndHousehold/*/personQualifiesTPforEitc`,
      },
    ],
  },
];

// These are used in the "you may qualify but you haven't yet" conditional list on qualified-dependent and confirmed-qp
// This is a list of all credits this person still may qualify for but not ones they have already qualified for.
const hasIncompletePossibleBenefitsItems: ItemConfig[] = [
  {
    itemKey: `hoh`,
    conditions: [
      { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/personQualifiesTPforHoH` },
      { operator: `isIncomplete`, condition: `/eligibleForHoh` },
    ],
  },
  {
    itemKey: `qss`,
    conditions: [
      { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/personQualifiesTPforQss` },
      { operator: `isIncomplete`, condition: `/eligibleForQss` },
    ],
  },
  {
    itemKey: `cdcc`,
    conditions: [
      {
        operator: `isTrueOrIncomplete`,
        condition: `/familyAndHousehold/*/personQualifiesTPforCdcc`,
      },
      { operator: `isIncomplete`, condition: `/cdccQualified` },
    ],
  },
  {
    itemKey: `ctc`,
    conditions: [
      { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/eligibleCtc` },
      { operator: `isIncomplete`, condition: `/ctcQualified` },
    ],
  },
  {
    itemKey: `odc`,
    conditions: [
      { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/eligibleOdc` },
      { operator: `isIncomplete`, condition: `/odcQualified` },
    ],
  },
  {
    itemKey: `eitc`,
    conditions: [
      {
        operator: `isTrueOrIncomplete`,
        condition: `/familyAndHousehold/*/personQualifiesTPforEitc`,
      },
      { operator: `isIncomplete`, condition: `/eitcQualified` },
    ],
  },
];

export const FamilyAndHHSubcategory = (
  <Subcategory
    route='dependents'
    completeIf='/familyAndHouseholdIsDone'
    lockFutureSectionsIfCollectionItemsIncomplete={true}
    collectionContext='/familyAndHousehold'
    dataItems={[
      {
        itemKey: `claimedDependents`,
        conditions: [`/hasClaimedDependents`],
      },
    ]}
  >
    <Gate condition={{ operator: `isFalse`, condition: `/filersCouldHaveDependentOrQualifyingPerson` }}>
      <Screen route='family-hh-intro-dep-tps' actAsDataView={true}>
        {/* TODO #4386 context header with no heading */}
        <Heading i18nKey='/heading/you-and-your-family/dependents/intro-dep-tps' />
        {/* This is a stopgap solution to resolve an issue where we weren't setting the familyAndHousehold fact to an
          empty array. This should be resolved as part of: 
          https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/15596.
        */}
        <SetFactAction
          path='/familyAndHousehold'
          source='emptyCollection'
          condition={{ operator: `isIncomplete`, condition: `/familyAndHousehold` }}
        />
        <SetFactAction path='/familyAndHouseholdIsDone' source='/flowTrue' />
        <IntroContent
          i18nKey='/info/you-and-your-family/dependents/intro-dep-tps'
          condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
        />
        <DFModal
          batches={[`cdcc-0`]}
          i18nKey='/info/you-and-your-family/dependents/intro-dep-tps'
          condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
        />
        <IntroContent i18nKey='/info/you-and-your-family/dependents/intro-dep-tps-mfj' condition='/treatAsMFJ' />
        <InternalLink
          i18nKey='/info/you-and-your-family/dependents/intro-dep-tps-mfj'
          route='/flow/you-and-your-family/spouse/spouse-mfj-refund-only'
          condition='/treatAsMFJ'
        />
        <SaveAndOrContinueButton />
      </Screen>
    </Gate>
    {/* This gate condition should never be incomplete by the time we reach dependents, but we have it here so that the section will appear in the checklist  */}
    <Gate condition={{ operator: `isTrueOrIncomplete`, condition: `/filersCouldHaveDependentOrQualifyingPerson` }}>
      <Screen route='family-hh-intro-non-dep-tps'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/you-and-your-family/dependents' />
        <Heading i18nKey='/heading/you-and-your-family/dependents/intro' />
        <IntroContent i18nKey='/info/you-and-your-family/dependents/intro' />
        <DFModal batches={[`cdcc-0`]} i18nKey='/info/you-and-your-family/dependents/intro' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='family-hh-intro-2'>
        <Heading i18nKey='/heading/you-and-your-family/dependents/intro-2' />
        <DFModal batches={[`cdcc-0`]} i18nKey='/info/you-and-your-family/dependents/intro-2' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='add-person'>
        <MefAlert mefErrorCode='IND-996' i18nKey='family-and-household' type='warning' />
        <MefAlert
          type='warning'
          mefErrorCode='IND-507'
          i18nKey='family-and-household'
          factPaths={[`/familyAndHousehold/*/fullName`]}
        />
        <Heading i18nKey='/heading/you-and-your-family/dependents/add' />
        <DFModal
          i18nKey='/info/you-and-your-family/dependents/add'
          condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
        />
        <DFModal i18nKey='/info/you-and-your-family/dependents/add-mfj' condition='/treatAsMFJ' />
        <CollectionItemManager
          path='/familyAndHousehold'
          loopName='/familyAndHousehold'
          donePath='/familyAndHouseholdIsDone'
        />
      </Screen>
      <CollectionLoop
        loopName='/familyAndHousehold'
        collection='/familyAndHousehold'
        iconName='Person'
        donePath='/familyAndHouseholdIsDone'
        collectionItemCompletedCondition='/familyAndHousehold/*/isCompleted'
        dataViewSections={[
          {
            i18nKey: `dataviews./flow/you-and-your-family/dependents.mayQualify`,
            i18nModalKey: `dataviews./flow/you-and-your-family/dependents.mayQualifyModal`,
            conditions: [
              `/familyAndHousehold/*/isCompleted`,
              { operator: `isFalse`, condition: `/familyAndHousehold/*/doesNotQualifyForAnyBenefits` },
            ],
            itemAssertions: [
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.success-incomplete-filing-and-credits.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.success-incomplete-filing-and-credits.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isIncomplete`, condition: `/filingStatus` },
                  { operator: `isFalseOrIncomplete`, condition: `/creditsSectionComplete` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  { operator: `isFalseOrIncomplete`, condition: `/creditsSectionComplete` },
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForStatuses` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-status.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-status.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  { operator: `isFalseOrIncomplete`, condition: `/creditsSectionComplete` },
                  `/familyAndHousehold/*/qualifiesForStatuses`,
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForStatuses` },
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForTaxCredits` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-status.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-status.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  `/familyAndHousehold/*/qualifiesForStatuses`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForTaxCredits` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-credits.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-credits.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForStatuses` },
                  `/familyAndHousehold/*/qualifiesForTaxCredits`,
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-status-and-credits.brief`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.base-with-status-and-credits.full`,
                subSubCategoryToEdit: `claim-choice`,
                type: `success`,
                conditions: [
                  `/familyAndHousehold/*/isClaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  `/familyAndHousehold/*/qualifiesForStatuses`,
                  `/familyAndHousehold/*/qualifiesForTaxCredits`,
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.warning-unclaimed-not-qp`,
                type: `warning`,
                conditions: [`/familyAndHousehold/*/isUnclaimedDependent`],
              },
              {
                // filing status complete base
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-unclaimed-qp-base.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-unclaimed-qp-base.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.success.brief`,
                subSubCategoryToEdit: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/isUnclaimedDependent`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isIncomplete`, condition: `/filingStatus` },
                  { operator: `isFalseOrIncomplete`, condition: `/creditsSectionComplete` },
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  { operator: `isFalseOrIncomplete`, condition: `/creditsSectionComplete` },
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForStatuses` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-status.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-status.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  { operator: `isFalseOrIncomplete`, condition: `/creditsSectionComplete` },
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                  `/familyAndHousehold/*/qualifiesForStatuses`,
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForStatuses` },
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForTaxCredits` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-status.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-status.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                  `/familyAndHousehold/*/qualifiesForStatuses`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForTaxCredits` },
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-credits.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-credits.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                  { operator: `isFalseOrIncomplete`, condition: `/familyAndHousehold/*/qualifiesForStatuses` },
                  `/familyAndHousehold/*/qualifiesForTaxCredits`,
                ],
              },
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-status-and-credits.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.info-eligible-non-dependent-base-with-status-and-credits.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-may-qualify`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-may-qualify`,
                outcomeReviewRoute: `claim-choice`,
                type: `info`,
                conditions: [
                  `/familyAndHousehold/*/doesNotQualify`,
                  { operator: `isComplete`, condition: `/filingStatus` },
                  `/creditsSectionComplete`,
                  `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                  `/familyAndHousehold/*/qualifiesForStatuses`,
                  `/familyAndHousehold/*/qualifiesForTaxCredits`,
                ],
              },
            ],
          },
          {
            i18nKey: `dataviews./flow/you-and-your-family/dependents.incomplete`,
            condition: { operator: `isTrueOrIncomplete`, condition: `/familyAndHousehold/*/isIncomplete` },
          },
          {
            i18nKey: `dataviews./flow/you-and-your-family/dependents.doNotQualify`,
            i18nModalKey: `dataviews./flow/you-and-your-family/dependents.doNotQualifyModal`,
            conditions: [`/familyAndHousehold/*/doesNotQualifyForAnyBenefits`],
            itemAssertions: [
              {
                i18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.inactive-does-not-qualify.brief`,
                collectionItemI18nKey: `dataviews./flow/you-and-your-family/dependents.assertions.inactive-does-not-qualify.full`,
                resultI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.brief-no-benefits`,
                outcomeI18nKey: `dataviews./flow/you-and-your-family/dependents.results.failure.full-no-benefits`,
                outcomeReviewRoute: `claim-choice`,
                type: `inactive`,
              },
            ],
          },
        ]}
      >
        <SubSubcategory route='basic-info'>
          {/*  Name, Age, Relationship  */}

          <Screen route='add-person-basic-info'>
            <MefAlert
              type='warning'
              mefErrorCode='IND-507'
              i18nKey='add-person-basic-info'
              factPaths={[`/familyAndHousehold/*/fullName`]}
              internalLink='/flow/you-and-your-family/dependents/qualified-dependent-tin-input'
            />
            <MefAlert
              mefErrorCode='SEIC-F1040-535-04'
              i18nKey='date-of-birth'
              factPaths={[`/familyAndHousehold/*/dateOfBirth`]}
              type='warning'
              internalLink='/flow/you-and-your-family/dependents/qualified-dependent-tin-input'
            />
            <MefAlert
              mefErrorCode='IND-116-01'
              i18nKey='date-of-birth'
              factPaths={[`/familyAndHousehold/*/dateOfBirth`]}
              type='warning'
              internalLink='/flow/you-and-your-family/dependents/qualified-dependent-tin-input'
            />
            <MefAlert
              type='warning'
              mefErrorCode='R0000-504-02'
              i18nKey='full-name'
              factPaths={[`/familyAndHousehold/*/fullName`]}
              internalLink='/flow/you-and-your-family/dependents/qualified-dependent-tin-input'
            />
            <MefAlert
              type='warning'
              mefErrorCode='SEIC-F1040-501-02'
              i18nKey='full-name'
              factPaths={[`/familyAndHousehold/*/fullName`]}
              internalLink='/flow/you-and-your-family/dependents/qualified-dependent-tin-input'
            />
            <Heading i18nKey='/heading/you-and-your-family/dependents/basic-info' />
            <InfoDisplay i18nKey='/info/you-and-your-family/dependents/intro-3' />
            <DFModal i18nKey='/info/you-and-your-family/dependents/what-if-they-changed-their-legal-name' />
            <LimitingString path='/familyAndHousehold/*/firstName' displayOnlyOn='edit' />
            <LimitingString path='/familyAndHousehold/*/writableMiddleInitial' displayOnlyOn='edit' required={false} />
            <LimitingString path='/familyAndHousehold/*/lastName' displayOnlyOn='edit' />
            <Enum path='/familyAndHousehold/*/writableSuffix' displayOnlyOn='edit' required={false} skipBlank={true} />
            <GenericString path='/familyAndHousehold/*/fullName' displayOnlyOn='data-view' />
            <DatePicker path='/familyAndHousehold/*/dateOfBirth' />
            <Boolean path='/familyAndHousehold/*/deceased' inputType='checkbox' required={false} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>

        <SubSubcategory route='relationship'>
          <Screen route='add-person-relationship-category'>
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/relationship-category'
              condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
            />
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/relationship-category-mfj'
              condition='/treatAsMFJ'
            />
            <InfoDisplay
              i18nKey='/info/you-and-your-family/dependents/relationship-description-mfj'
              condition='/treatAsMFJ'
            />
            <DFModal i18nKey='/info/you-and-your-family/dependents/relationship-category-ancestors' />
            <DFModal i18nKey='/info/you-and-your-family/dependents/relationship-category-separated-divorced-died' />
            <Enum path='/familyAndHousehold/*/relationshipCategory' />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='add-person-relationship-type-child'
            condition='/familyAndHousehold/*/relationshipCategoryIsChild'
          >
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/choose-relationship'
              condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
            />
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/choose-relationship-mfj'
              condition='/treatAsMFJ'
            />
            <InfoDisplay
              i18nKey='/info/you-and-your-family/dependents/relationship-description-mfj-child'
              condition='/treatAsMFJ'
            />
            <DFModal i18nKey='/info/you-and-your-family/dependents/relationship-type-child' />
            <Enum path='/familyAndHousehold/*/childRelationship' />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='add-person-relationship-type-sibling'
            condition='/familyAndHousehold/*/relationshipCategoryIsSibling'
          >
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/choose-relationship'
              condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
            />
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/choose-relationship-mfj'
              condition='/treatAsMFJ'
            />
            <InfoDisplay
              i18nKey='/info/you-and-your-family/dependents/relationship-description-mfj-sibling'
              condition='/treatAsMFJ'
            />
            <DFModal i18nKey='/info/you-and-your-family/dependents/relationship-type-sibling' />
            <Enum path='/familyAndHousehold/*/siblingRelationship' renderAs='radio' />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='add-person-relationship-type-parent'
            condition='/familyAndHousehold/*/relationshipCategoryIsParent'
          >
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/choose-relationship'
              condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
            />
            <Heading
              i18nKey='/heading/you-and-your-family/dependents/choose-relationship-mfj'
              condition='/treatAsMFJ'
            />
            <InfoDisplay
              i18nKey='/info/you-and-your-family/dependents/relationship-description-mfj-parent'
              condition='/treatAsMFJ'
            />
            <DFModal i18nKey='/info/you-and-your-family/dependents/relationship-type-parent' />
            <Enum path='/familyAndHousehold/*/parentalRelationship' />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='add-person-relationship-type-in-law'
            condition='/familyAndHousehold/*/relationshipCategoryIsInlaw'
          >
            <Heading i18nKey='/heading/you-and-your-family/dependents/choose-relationship' />
            <InfoDisplay
              i18nKey='/info/you-and-your-family/dependents/relationship-description-mfj-in-law'
              condition='/treatAsMFJ'
            />
            <Enum path='/familyAndHousehold/*/inlawRelationship' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>

        {/*  Student, Disability, Self-care  */}

        <Gate condition='/familyAndHousehold/*/studentStatusMayAffectBenefits'>
          <SubSubcategory route='life-circumstances'>
            <Screen route='add-person-student'>
              <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-student' />
              <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-student-full-time' />
              <Boolean path='/familyAndHousehold/*/fullTimeStudent' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
        </Gate>
        <SubSubcategory route='life-circumstances'>
          <Gate condition='/familyAndHousehold/*/disabilityStatusMayAffectBenefits'>
            <Screen route='add-person-disability'>
              <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-disability' />
              <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-disability-what-is-permanent' />
              <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-disability-what-is-substantial-employment' />
              <Boolean path='/familyAndHousehold/*/permanentTotalDisability' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Screen route='add-person-self-care'>
            <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-self-care' batches={[`cdcc-0`]} />
            <DFModal i18nKey='/info/you-and-your-family/dependents/self-care' />
            <Boolean path='/familyAndHousehold/*/unableToCareForSelf' batches={[`cdcc-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>

        {/*  Residency duration  */}

        <SubSubcategory route='residency'>
          <Screen route='add-person-lived-with-you'>
            <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-lived-with-tp' />
            <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-lived-with-tp' />

            <DFModal
              i18nKey='/info/you-and-your-family/dependents/add-person-lived-with-tp-birth-adoption-kidnap-death'
              condition='/familyAndHousehold/*/mayHaveNotlivedWithTaxpayerInUSForFullTaxYear'
              items={[
                { itemKey: `conditionOne`, conditions: [`/familyAndHousehold/*/yearOfBirthSameAsTaxYear`] },
                { itemKey: `conditionTwo`, conditions: [`/familyAndHousehold/*/childIsFosterOrAdopted`] },
                { itemKey: `conditionThree`, conditions: [`/familyAndHousehold/*/youngerThanEighteenAtEndOfYear`] },
                { itemKey: `conditionFour`, conditions: [`/familyAndHousehold/*/deceased`] },
              ]}
            />
            <Enum path='/familyAndHousehold/*/residencyDuration' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>

        {/*  Special rule benefit split  */}

        <SubSubcategory route='parents-guardians'>
          {/*  Mural diamond 1 (logic reversed to test for entry rather than skipping)  */}
          <Gate condition='/familyAndHousehold/*/childMayQualifyForBenefitSplit'>
            {/*  Mural diamond 2 for people with relationship "biological child" or "adopted child"  */}
            <Gate condition='/familyAndHousehold/*/tpIsParent'>
              <Screen route='add-person-another-parent-in-picture'>
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-another-parent-in-picture-non-mfj'
                  condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                />
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-another-parent-in-picture-mfj'
                  condition='/treatAsMFJ'
                />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-another-parent-in-picture' />
                <Boolean path='/familyAndHousehold/*/hasOtherBiologicalOrAdoptiveParent' />
                <SaveAndOrContinueButton />
              </Screen>

              <Gate condition='/familyAndHousehold/*/hasOtherBiologicalOrAdoptiveParent'>
                <Screen route='add-person-other-parent-written-declaration'>
                  <Heading
                    condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                    i18nKey='/heading/you-and-your-family/dependents/add-person-other-parent-written-declaration'
                  />
                  <Heading
                    condition='/treatAsMFJ'
                    i18nKey='/heading/you-and-your-family/dependents/add-person-other-parent-written-declaration-mfj'
                  />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-other-parent-written-declaration' />
                  <DFModal i18nKey='/info/you-and-your-family/dependents/what-is-form-8332-snack' />
                  <Enum
                    condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                    i18nKeySuffixContext='self'
                    path='/familyAndHousehold/*/whichParentNotClaiming'
                  />
                  <Enum
                    condition='/treatAsMFJ'
                    i18nKeySuffixContext='mfj'
                    path='/familyAndHousehold/*/whichParentNotClaiming'
                  />
                  <SaveAndOrContinueButton />
                </Screen>

                <Gate condition='/familyAndHousehold/*/parentalSomeParentNotClaiming'>
                  <Screen route='add-person-written-declaration-signed'>
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-written-declaration-signed' />
                    <InfoDisplay
                      condition='/familyAndHousehold/*/primaryFilerNotClaiming'
                      i18nKey='/info/you-and-your-family/dependents/add-person-written-declaration-signed-primary-filer'
                    />
                    <InfoDisplay
                      condition='/familyAndHousehold/*/bothFilersNotClaiming'
                      i18nKey='/info/you-and-your-family/dependents/add-person-written-declaration-signed-both-filers'
                    />
                    <InfoDisplay
                      condition='/familyAndHousehold/*/otherParentNotClaiming'
                      i18nKey='/info/you-and-your-family/dependents/add-person-written-declaration-signed-other-parent'
                    />
                    <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-written-declaration-signed' />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='add-person-not-in-custody'>
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-not-in-custody' />
                    <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-not-in-custody' />
                    <DFModal i18nKey='/info/you-and-your-family/dependents/majority-status-and-custody-snack' />
                    <Boolean path='/familyAndHousehold/*/inParentsCustody' i18nKeySuffixContext='ours' />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Gate condition='/familyAndHousehold/*/inParentsCustody'>
                    <Screen route='add-person-special-rule-residency'>
                      <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-residency' />
                      <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-residency' />
                      <DFModal i18nKey='/info/you-and-your-family/dependents/night-with-parent-snack' />
                      <Boolean path='/familyAndHousehold/*/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths' />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Gate condition='/familyAndHousehold/*/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths'>
                      <Screen route='add-person-special-rule-nights-together'>
                        <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-nights-with-tp-vs-other-parent' />
                        <DFModal i18nKey='/info/you-and-your-family/dependents/night-with-parent-snack' />
                        <Enum path='/familyAndHousehold/*/nightsWithTpVsOtherParent' />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen
                        route='add-person-special-rule-agi-tiebreaker'
                        condition='/familyAndHousehold/*/hadEqualNumberOfNightsWithParents'
                      >
                        <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-tp-agi-higher-than-other-parent' />
                        <DFModal i18nKey='/info/you-and-your-family/dependents/what-is-agi-snack' />
                        <Boolean path='/familyAndHousehold/*/tpAgiHigherThanOtherParent' />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Screen route='add-person-special-rule-living-marital-status'>
                        <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-living-marital-status' />
                        <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-living-marital-status' />
                        <Enum path='/familyAndHousehold/*/parentalSituation' />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Gate condition='/familyAndHousehold/*/parentsAreDivorcedSeparatedOrLivingApart'>
                        <Screen route='add-person-special-rule-support'>
                          <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-support' />
                          <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-support' />
                          <Boolean path='/familyAndHousehold/*/parentalSupport' />
                          <SaveAndOrContinueButton />
                        </Screen>
                        <Gate condition='/familyAndHousehold/*/parentalSupport'>
                          {/* We did it! We made it and are eligible for a benefit split. We just apply who is custodial to see which
                              way the benefits are split */}
                          <Screen
                            route='add-person-special-rule-applies-custodial'
                            condition='/familyAndHousehold/*/tpCanClaimSplitBenefitsHohEitcCdcc'
                          >
                            <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-applies-custodial' />
                            <InfoDisplay
                              batches={[`cdcc-0`]}
                              i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-applies-custodial'
                            />
                            <SaveAndOrContinueButton />
                          </Screen>
                          {/* If they are not the custodial parent, they are knocked out!
                              see the fact /knockedOutByNonCustodialParentBenefitSplit

                              Similarly if they are the custodial parent, but the other parent
                              submitted form 8832, they are knocked out with /knockedOutByContradictory8832

                              Those screens run directly below these comments
                          */}
                        </Gate>
                      </Gate>
                    </Gate>
                  </Gate>
                  <Screen
                    route='add-person-special-rule-applies-non-custodial'
                    condition='/knockedOutByNonCustodialParentBenefitSplit'
                    isKnockout={true}
                  >
                    <IconDisplay name='ErrorOutline' size={9} isCentered />
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-applies-non-custodial' />
                    <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-applies-non-custodial' />
                    <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
                    <KnockoutButton i18nKey='button.knockout' />
                  </Screen>
                  <Screen
                    route='add-person-special-rule-conflicting-information'
                    condition='/knockedOutByContradictory8832'
                    isKnockout={true}
                  >
                    <IconDisplay name='ErrorOutline' size={9} isCentered />
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-contradicting-8832' />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-contradicting-8832-they-did'
                      condition='/contradictory8832TPIsCustodialParentTriggeredKnockout'
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-contradicting-8832-i-did'
                      condition='/contradictory8832TPNotCustodialParentTriggeredKnockout'
                    />
                    <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
                    <KnockoutButton i18nKey='button.knockout' />
                  </Screen>

                  <Screen
                    route='add-person-special-rule-does-not-apply'
                    condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/eligibleForBenefitSplit` }}
                  >
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-does-not-apply' />
                    <InfoDisplay
                      condition='/familyAndHousehold/*/primaryFilerNotClaiming'
                      i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-does-not-apply-primary-filer'
                    />
                    <InfoDisplay
                      condition='/familyAndHousehold/*/bothFilersNotClaiming'
                      i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-does-not-apply-both-filers'
                    />
                    <InfoDisplay
                      condition='/familyAndHousehold/*/otherParentNotClaiming'
                      i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-does-not-apply-other-parent'
                    />
                    <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-does-not-apply' />
                    <SaveAndOrContinueButton />
                  </Screen>
                </Gate>
              </Gate>
            </Gate>

            {/*  Mural diamond 3 for a person who is not the taxpayers' biological or adopted child */}
            <Gate condition='/familyAndHousehold/*/tpIsNotParent'>
              <Screen route='add-person-not-in-custody-other-rel'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-not-in-custody-other-rel' />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-not-in-custody-other-rel' />
                <DFModal i18nKey='/info/you-and-your-family/dependents/majority-status-and-custody-snack' />
                <Boolean path='/familyAndHousehold/*/inParentsCustody' i18nKeySuffixContext='theirs' />
                <SaveAndOrContinueButton />
              </Screen>

              <Gate condition='/familyAndHousehold/*/inParentsCustody'>
                <Screen route='add-person-parents-living'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-parents-living' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-parents-living' />
                  <Boolean path='/familyAndHousehold/*/biologicalOrAdoptiveParentsLiving' />
                  <SaveAndOrContinueButton />
                </Screen>
                <Gate condition='/familyAndHousehold/*/biologicalOrAdoptiveParentsLiving'>
                  <Screen route='add-person-any-parents-written-declaration'>
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-parent-not-claiming' />
                    <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-parent-not-claiming' />
                    <DFModal i18nKey='/info/you-and-your-family/dependents/what-is-form-8332-snack' />
                    <Boolean path='/familyAndHousehold/*/nonParentalSomeParentNotClaiming' />
                    <SaveAndOrContinueButton />
                  </Screen>

                  <Gate condition='/familyAndHousehold/*/nonParentalSomeParentNotClaiming'>
                    <Screen route='add-person-special-rule-question'>
                      <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-question' />
                      <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-question' />
                      <Boolean path='/familyAndHousehold/*/parentsMeetReqsRuleForChildrenOfDivorcedParents' />
                      <SaveAndOrContinueButton />
                    </Screen>

                    <Gate condition='/familyAndHousehold/*/parentsMeetReqsRuleForChildrenOfDivorcedParents'>
                      {/* We know the special rule applies for the child's biological or adoptive parents */}
                      <Screen route='add-person-special-rule-using-child'>
                        <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-using-child' />
                        <InfoDisplay
                          batches={[`cdcc-0`]}
                          i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-using-child'
                        />
                        <DFModal i18nKey='/info/you-and-your-family/dependents/who-is-the-custodial-parent-snack' />
                        <DFModal
                          batches={[`cdcc-0`]}
                          i18nKey='/info/you-and-your-family/dependents/why-need-know-if-using-child-snack'
                        />
                        <Boolean path='/familyAndHousehold/*/specialRuleChildUsedByCustodialParentAsQP' />
                        <SaveAndOrContinueButton />
                      </Screen>
                      <Gate
                        condition={{
                          operator: `isFalse`,
                          condition: `/familyAndHousehold/*/specialRuleChildUsedByCustodialParentAsQP`,
                        }}
                      >
                        <Screen route='add-person-special-rule-agi-test'>
                          <Heading
                            condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                            i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-agi-test'
                          />
                          <Heading
                            condition='/treatAsMFJ'
                            i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-agi-test-mfj'
                          />
                          <DFModal i18nKey='/info/you-and-your-family/dependents/what-is-agi-snack' />
                          <DFModal i18nKey='/info/you-and-your-family/dependents/who-is-the-custodial-parent-snack' />
                          <Boolean
                            path='/familyAndHousehold/*/specialRuleAgiTest'
                            i18nKeySuffixContext='self'
                            condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                          />
                          <Boolean
                            path='/familyAndHousehold/*/specialRuleAgiTest'
                            i18nKeySuffixContext='mfj'
                            condition='/treatAsMFJ'
                          />
                          <SaveAndOrContinueButton />
                        </Screen>

                        <Gate condition='/familyAndHousehold/*/specialRuleAgiTest'>
                          <Screen route='add-person-special-rule-other-eligible-tp-outcome'>
                            <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-special-rule-other-eligible-tp-outcome' />
                            <InfoDisplay
                              batches={[`cdcc-0`]}
                              i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-other-eligible-tp-outcome'
                              condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                            />
                            <InfoDisplay
                              i18nKey='/info/you-and-your-family/dependents/add-person-special-rule-other-eligible-tp-outcome-mfj'
                              condition='/treatAsMFJ'
                            />
                            <SaveAndOrContinueButton />
                          </Screen>
                        </Gate>
                      </Gate>
                    </Gate>
                  </Gate>
                </Gate>
              </Gate>
            </Gate>
          </Gate>
        </SubSubcategory>

        <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/disqualifiedDueToParentalCustody` }}>
          {/*  Person's home for the year  */}

          <SubSubcategory route='residency'>
            <Gate condition='/familyAndHousehold/*/flowAddPersonLivedWithTpInUs'>
              <Screen route='add-person-lived-with-you-in-us'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-lived-with-tp-in-us' />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-lived-with-tp-in-us' />
                <DFModal
                  i18nKey='/info/you-and-your-family/dependents/add-person-lived-with-tp-birth-adoption-kidnap-death-in-us'
                  condition='/familyAndHousehold/*/mayHaveNotlivedWithTaxpayerInUSForFullTaxYear'
                  items={[
                    { itemKey: `conditionOne`, conditions: [`/familyAndHousehold/*/yearOfBirthSameAsTaxYear`] },
                    { itemKey: `conditionTwo`, conditions: [`/familyAndHousehold/*/childIsFosterOrAdopted`] },
                    { itemKey: `conditionThree`, conditions: [`/familyAndHousehold/*/youngerThanEighteenAtEndOfYear`] },
                    { itemKey: `conditionFour`, conditions: [`/familyAndHousehold/*/deceased`] },
                  ]}
                />
                <Enum path='/familyAndHousehold/*/monthsLivedWithTPInUS' />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>

            <Gate condition='/familyAndHousehold/*/flowAddPersonKeepingUpHome'>
              <Screen route='add-person-keeping-up-home'>
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-keeping-up-home'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-keeping-up-home-mfj'
                  condition='/treatAsMFJ'
                />
                <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-keeping-up-home' />
                <Boolean
                  path='/familyAndHousehold/*/tpPaidMostOfHomeUpkeep'
                  i18nKeySuffixContext='i'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <Boolean
                  path='/familyAndHousehold/*/tpPaidMostOfHomeUpkeep'
                  i18nKeySuffixContext='we'
                  condition='/treatAsMFJ'
                />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>

            <Gate condition='/familyAndHousehold/*/flowAddPersonKeepingUpParentsHome'>
              <Screen route='add-person-keeping-up-parents-home'>
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-keeping-up-parents-home'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-keeping-up-parents-home-mfj'
                  condition='/treatAsMFJ'
                />
                <InfoDisplay
                  i18nKey='/info/you-and-your-family/dependents/add-person-keeping-up-parents-home'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <InfoDisplay
                  i18nKey='/info/you-and-your-family/dependents/add-person-keeping-up-parents-home-mfj'
                  condition='/treatAsMFJ'
                />
                <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-keeping-up-home' />
                <Boolean
                  path='/familyAndHousehold/*/tpPaidMostOfParentHomeUpkeep'
                  i18nKeySuffixContext='i'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <Boolean
                  path='/familyAndHousehold/*/tpPaidMostOfParentHomeUpkeep'
                  i18nKeySuffixContext='we'
                  condition='/treatAsMFJ'
                />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
          </SubSubcategory>

          {/*  TIN test  */}
          <SubSubcategory route='tax-id'>
            <Screen route='add-person-tin'>
              <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-tin' />
              <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-tin' />
              <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-tin1' />
              <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-tin2' />
              <Enum path='/familyAndHousehold/*/tinType' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='add-personhoh-qp-no-tin' condition='/knockoutHohQpHasNoTin' isKnockout={true}>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/hoh-qp-no-tin' />
              <InfoDisplay i18nKey='/info/knockout/hoh-qp-no-tin' />
              <DFAlert i18nKey='/info/knockout/how-to-file-paper' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Screen
              route='add-person-born-died-no-tin'
              condition='/knockoutPersonBornAndDiedInTaxYearWithoutTin'
              isKnockout={true}
            >
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/add-person-born-died-no-tin' />
              <InfoDisplay i18nKey='/info/knockout/add-person-born-died-no-tin' />
              <DFModal i18nKey='/info/knockout/add-person-born-died-no-tin' />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
            <Gate condition='/familyAndHousehold/*/flowAddPersonSsnValidForWork'>
              <Screen route='add-person-ssn-valid-for-work'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-ssn-valid-for-work' />
                <DFModal i18nKey='/info/you-and-your-family/ssn-valid-for-work/where' />
                <DFModal i18nKey='/info/you-and-your-family/ssn-valid-for-work/needs-to-be-updated-spouse' />
                <Enum path='/familyAndHousehold/*/ssnEmploymentValidity' />
                <SaveAndOrContinueButton />
              </Screen>

              <Screen
                route='add-person-ssn-federal-benefits'
                condition='/familyAndHousehold/*/ssnNotValidForEmployment'
              >
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-ssn-federal-benefits' />
                <Boolean path='/familyAndHousehold/*/ssnOnlyForFederallyFundedBenefit' />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>

            <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/hasTinTypeOfNone` }}>
              <Screen route='add-person-acknowledge-tin'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-acknowledge-tin' />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-acknowledge-tin' />
                <Tin
                  path='/familyAndHousehold/*/tin'
                  displayOnlyOn='data-view'
                  dataViewAnchorLink='claim-choice'
                  isSensitive={true}
                />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
          </SubSubcategory>

          {/*  Support tests  */}
          <SubSubcategory route='support'>
            <Gate condition='/familyAndHousehold/*/flowShowQcSupportTest'>
              <Screen route='add-person-qc-support'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-qc-support' />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-qc-support' />
                <Boolean path='/familyAndHousehold/*/ownSupport' />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>

            <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/flowShowQcSupportTest` }}>
              <Screen route='add-person-qr-support'>
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-qr-support'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-qr-support-mfj'
                  condition='/treatAsMFJ'
                />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-qr-support' />
                <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-multiple-support' />
                <Boolean
                  path='/familyAndHousehold/*/writableQrSupportTest'
                  i18nKeySuffixContext='i'
                  condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
                />
                <Boolean
                  path='/familyAndHousehold/*/writableQrSupportTest'
                  i18nKeySuffixContext='we'
                  condition='/treatAsMFJ'
                />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='add-person-qr-gross-income'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-qr-gross-income' />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-qr-gross-income' />
                <DFModal i18nKey='/info/you-and-your-family/dependents/sheltered-employment' />
                <Boolean path='/familyAndHousehold/*/grossIncomeTest' />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
          </SubSubcategory>

          {/*  Marital status + joint return test  */}
          <SubSubcategory route='marital-status'>
            <Screen route='add-person-married'>
              <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-married' />
              <Boolean path='/familyAndHousehold/*/married' />
              <SaveAndOrContinueButton />
            </Screen>

            <Gate condition='/familyAndHousehold/*/married'>
              <Screen route='add-person-joint-return'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-joint-return' />
                <Boolean path='/familyAndHousehold/*/writableJointReturn' />
                <SaveAndOrContinueButton />
              </Screen>
              <Gate condition='/familyAndHousehold/*/jointReturn'>
                <Screen route='add-person-filing-requirement'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-filing-requirement' />
                  <DFModal i18nKey='/info/you-and-your-family/dependents/how-to-know-required-to-file' />
                  <Boolean path='/familyAndHousehold/*/writableRequiredToFile' />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen
                  route='add-person-filing-for-refund-only'
                  condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/writableRequiredToFile` }}
                >
                  <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-filing-for-refund-only' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-filing-for-refund-only' />
                  <Boolean path='/familyAndHousehold/*/writableFilingOnlyForRefund' />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>
            </Gate>
          </SubSubcategory>

          <SubSubcategory route='citizenship'>
            <Screen route='add-person-us-citizen'>
              <Heading i18nKey='/heading/you-and-your-family/dependents/citizenship' />
              <DFModal i18nKey='/info/why-citizenship' />
              <InfoDisplay
                i18nKey='/info/you-and-your-family/dependents/citizenship-adopted'
                conditions={[`/familyAndHousehold/*/childIsAdopted`, `/eitherFilerUSCitizenOrNational`]}
              />
              <Boolean path='/familyAndHousehold/*/isUsCitizenFullYear' />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/isUsCitizenFullYear` }}>
              <Screen route='add-person-resident-alien'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/resident' />
                <DFModal i18nKey='/info/you-and-your-family/dependents/resident' />
                <Boolean path='/familyAndHousehold/*/writableUsResident' />
                <SaveAndOrContinueButton />
              </Screen>
              <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/usResident` }}>
                <Screen route='add-person-us-national'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/national' />
                  <DFModal i18nKey='/info/you-and-your-family/dependents/national' />
                  <Boolean path='/familyAndHousehold/*/writableUsNational' />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen
                  route='add-person-resident-mexico-canada'
                  condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/usNational` }}
                >
                  <Heading i18nKey='/heading/you-and-your-family/dependents/cnmx' />
                  <Boolean path='/familyAndHousehold/*/writableCanadaMexicoResident' />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>
            </Gate>
          </SubSubcategory>

          {/*  Qualifying child of another  */}
          <SubSubcategory route='qc-of-another'>
            <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/couldntBeQCOfAnother` }}>
              <Screen route='add-person-qc-of-another'>
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-qc-of-another'
                  condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
                />
                <Heading
                  i18nKey='/heading/you-and-your-family/dependents/add-person-qc-of-another-mfj'
                  condition='/treatAsMFJ'
                />
                <DFModal i18nKey='/info/you-and-your-family/dependents/add-person-qc-of-another-snack' />
                <Boolean path='/familyAndHousehold/*/writableCouldBeQualifyingChildOfAnother' />
                <SaveAndOrContinueButton />
              </Screen>

              <Gate condition='/familyAndHousehold/*/qrPathCouldBeQualifyingChildOfAnother'>
                <Screen route='add-person-qc-of-another-claimer-filing-requirement'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-qc-of-another-claimer-filing-requirement' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-qc-of-another-claimer-filing-requirement' />
                  <DFModal i18nKey='/info/you-and-your-family/dependents/required-to-file-snack' />
                  <Boolean path='/familyAndHousehold/*/writablePotentialClaimerMustFile' />
                  <SaveAndOrContinueButton />
                </Screen>

                <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/potentialClaimerMustFile` }}>
                  <Screen route='add-person-qc-of-another-claimer-filing'>
                    <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-qc-of-another-claimer-filing' />
                    <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-qc-of-another-claimer-filing' />
                    <Boolean path='/familyAndHousehold/*/writablePotentialClaimerDidFile' />
                    <SaveAndOrContinueButton />
                  </Screen>

                  <Gate condition='/familyAndHousehold/*/potentialClaimerDidFile'>
                    <Screen route='add-person-qc-of-another-claimers-return'>
                      <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-qc-of-another-claimers-return' />
                      <InfoDisplay i18nKey='/info/you-and-your-family/dependents/add-person-qc-of-another-claimers-return' />
                      <Boolean path='/familyAndHousehold/*/writablePotentialClaimerFiledOnlyForRefund' />
                      <SaveAndOrContinueButton />
                    </Screen>
                  </Gate>
                </Gate>
              </Gate>
            </Gate>
            <Screen route='add-personhoh-qp-no-tin-end' condition='/knockoutHohQpHasNoTin' isKnockout={true}>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/hoh-qp-no-tin' />
              <InfoDisplay i18nKey='/info/knockout/hoh-qp-no-tin' />
              <DFAlert i18nKey='/info/knockout/how-to-file-paper' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>

            <Gate condition='/familyAndHousehold/*/qcOfAnotherButNot'>
              <Screen route='add-person-qc-of-another-summary'>
                <Heading i18nKey='/heading/you-and-your-family/dependents/add-person-qc-of-another-summary' />
                <InfoDisplay
                  i18nKey='/info/you-and-your-family/dependents/you-and-your-family/dependents/add-person-qc-of-another-summary-did-file'
                  condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/potentialClaimerDidFile` }}
                />
                <InfoDisplay
                  i18nKey='/info/you-and-your-family/dependents/you-and-your-family/dependents/add-person-qc-of-another-summary-filed-only-for-refund'
                  condition='/familyAndHousehold/*/potentialClaimerFiledOnlyForRefund'
                />
                <InfoDisplay i18nKey='/info/you-and-your-family/dependents/you-and-your-family/dependents/add-person-qc-of-another-summary' />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
          </SubSubcategory>

          <SubSubcategory route='claim-choice'>
            {/*  Determination + choice  */}
            {/* Person is an eligible dependent*/}
            <Gate condition='/familyAndHousehold/*/eligibleDependent'>
              {/* Child can be claimed by multiple people*/}
              <Gate condition='/familyAndHousehold/*/isQualifyingChildOfAnother'>
                <Screen route='qualified-qc-of-multiple-tps'>
                  <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-qc-of-multiple-tps' />
                  <DFModal i18nKey='/info/you-and-your-family/dependents/qualified-qc-of-multiple-tps' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/qualified-qc-of-multiple-tps' />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen route='qualified-qc-of-multiple-tps-choice'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-qc-of-multiple-tps-choice' />
                  <DFAlert
                    i18nKey='/info/you-and-your-family/dependents/qualified-qc-of-multiple-tps-choice'
                    headingLevel='h2'
                    type='warning'
                  >
                    <DFModal i18nKey='/info/you-and-your-family/dependents/qualified-qc-of-multiple-tps-choice-modal'></DFModal>
                  </DFAlert>
                  <FactResultAssertion
                    i18nKey='dataviews./flow/you-and-your-family/dependents.results.success.full-another-tp-claims'
                    displayOnlyOn='data-view'
                    conditions={[
                      `/familyAndHousehold/*/isCompleted`,
                      { operator: `isFalse`, condition: `/familyAndHousehold/*/doesNotQualify` },
                    ]}
                  />
                  <Boolean path='/familyAndHousehold/*/tpClaims' />
                  <FactAssertion
                    type='warning'
                    i18nKey='dataviews./flow/you-and-your-family/dependents.alerts.claim-choice'
                    conditions={[`/familyAndHousehold/*/isUnclaimedDependent`]}
                    displayOnlyOn='data-view'
                  />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>

              <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/isQualifyingChildOfAnother` }}>
                <Screen route='qualified-dependent'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-dependent' />
                  {/* Qualification not known for any benefit yet */}
                  <DFModal
                    i18nKey='/info/you-and-your-family/dependents/qualified-dependent'
                    condition={{
                      operator: `isFalseOrIncomplete`,
                      condition: `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                    }}
                  />
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/dependents/qualified-dependent'
                    batches={[`cdcc-2`]}
                    conditions={[
                      {
                        operator: `isTrueOrIncomplete`,
                        condition: `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                      },
                      {
                        operator: `isFalseOrIncomplete`,
                        condition: `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                      },
                    ]}
                  />
                  {/* In the ConditionalList, these are all trueOrIncomplete as the claim choice hasn't been made yet, which is necessary to complete each of these qualifications. */}
                  <ConditionalList
                    i18nKey='/info/you-and-your-family/dependents/benefits-list'
                    batches={[`cdcc-2`]}
                    conditions={[
                      {
                        operator: `isTrueOrIncomplete`,
                        condition: `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                      },
                      {
                        operator: `isFalseOrIncomplete`,
                        condition: `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                      },
                    ]}
                    items={mayQualifyTPForTaxBenefitsItems}
                  />
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/dependents/qualified-dependent-note'
                    conditions={[
                      {
                        operator: `isTrueOrIncomplete`,
                        condition: `/familyAndHousehold/*/mayQualifyTPForTaxBenefits`,
                      },
                      {
                        operator: `isFalseOrIncomplete`,
                        condition: `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                      },
                    ]}
                  />

                  {/* Qualification status is known for at least one benefit */}
                  {/* YES benefits list: */}
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/dependents/qualified-dependent-yes-benefits'
                    batches={[`cdcc-2`]}
                    condition='/familyAndHousehold/*/doesQualifyTPForTaxBenefits'
                  />
                  <ConditionalList
                    i18nKey='/info/you-and-your-family/dependents/benefits-list'
                    batches={[`cdcc-2`]}
                    condition='/familyAndHousehold/*/doesQualifyTPForTaxBenefits'
                    items={[
                      {
                        itemKey: `hoh`,
                        conditions: [`/familyAndHousehold/*/personQualifiesTPforHoH`, `/eligibleForHoh`],
                      },
                      {
                        itemKey: `qss`,
                        conditions: [`/familyAndHousehold/*/personQualifiesTPforQss`, `/eligibleForQss`],
                      },
                      {
                        itemKey: `cdcc`,
                        conditions: [`/familyAndHousehold/*/cdccQualifyingPerson`, `/cdccQualified`],
                      },
                      {
                        itemKey: `ctc`,
                        conditions: [`/familyAndHousehold/*/eligibleCtc`, `/ctcQualified`],
                      },
                      {
                        itemKey: `odc`,
                        conditions: [`/familyAndHousehold/*/eligibleOdc`, `/odcQualified`],
                      },
                      {
                        itemKey: `eitc`,
                        conditions: [`/familyAndHousehold/*/eitcQualifyingChild`, `/eitcQualified`],
                      },
                    ]}
                  />
                  {/* MAYBE benefits list: */}
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/dependents/qualified-dependent-maybe-benefits'
                    batches={[`cdcc-2`]}
                    conditions={[
                      `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                      `/familyAndHousehold/*/hasIncompletePossibleBenefits`,
                    ]}
                  />
                  <ConditionalList
                    i18nKey='/info/you-and-your-family/dependents/benefits-list'
                    batches={[`cdcc-2`]}
                    conditions={[
                      `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                      `/familyAndHousehold/*/hasIncompletePossibleBenefits`,
                    ]}
                    items={hasIncompletePossibleBenefitsItems}
                  />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen route='qualified-dependent-choice'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-dependent-choice' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/qualified-dependent-choice' />
                  <DFModal
                    i18nKey='/info/you-and-your-family/dependents/can-another-taxpayer-claim-child'
                    condition='/familyAndHousehold/*/flowShowParentHelpText'
                  />
                  <FactResultAssertion
                    i18nKey='dataviews./flow/you-and-your-family/dependents.results.success.full'
                    displayOnlyOn='data-view'
                    conditions={[
                      `/familyAndHousehold/*/isCompleted`,
                      { operator: `isFalse`, condition: `/familyAndHousehold/*/doesNotQualify` },
                    ]}
                  />
                  <Boolean path='/familyAndHousehold/*/tpClaims' />
                  <FactAssertion
                    type='warning'
                    i18nKey='dataviews./flow/you-and-your-family/dependents.alerts.claim-choice'
                    conditions={[`/familyAndHousehold/*/isUnclaimedDependent`]}
                    displayOnlyOn='data-view'
                  />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>

              <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/tpClaims` }}>
                <Screen route='qualified-dependent-not-claimed'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-dependent-not-claimed' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/qualified-dependent-not-claimed' />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>

              {/*  Confirmation, TIN, PIN */}
              <Gate condition='/familyAndHousehold/*/isClaimedDependent'>
                <Screen route='qualified-dependent-tin-input'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-dependent-tin-input' />
                  <TaxReturnAlert
                    type='error'
                    i18nKey='/info/you-and-your-family/dependents/invalid-ssn'
                    conditions={[
                      `/familyAndHousehold/*/hasInvalidSSN`,
                      {
                        operator: `isFalseOrIncomplete`,
                        condition: `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                      },
                    ]}
                  />
                  <TaxReturnAlert
                    type='error'
                    i18nKey='/info/you-and-your-family/dependents/invalid-itin'
                    conditions={[
                      `/familyAndHousehold/*/hasInvalidITIN`,
                      {
                        operator: `isFalseOrIncomplete`,
                        condition: `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                      },
                    ]}
                  />
                  <TaxReturnAlert
                    type='error'
                    i18nKey='/info/you-and-your-family/dependents/invalid-atin'
                    conditions={[
                      `/familyAndHousehold/*/hasInvalidATIN`,
                      {
                        operator: `isFalseOrIncomplete`,
                        condition: `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                      },
                    ]}
                  />
                  <TaxReturnAlert
                    type='error'
                    i18nKey='/info/non-unique-tin'
                    conditions={[
                      { operator: `isComplete`, condition: `/familyAndHousehold/*/tin` },
                      { operator: `isFalse`, condition: `/familyAndHousehold/*/isTinUnique` },
                    ]}
                  />
                  <MefAlert
                    mefErrorCode='SEIC-F1040-535-04'
                    i18nKey='tin'
                    type='warning'
                    internalLink='/flow/you-and-your-family/dependents/add-person-basic-info'
                  />
                  <MefAlert
                    mefErrorCode='IND-116-01'
                    i18nKey='tin'
                    type='warning'
                    internalLink='/flow/you-and-your-family/dependents/add-person-basic-info'
                  />
                  <MefAlert
                    type='warning'
                    mefErrorCode='R0000-504-02'
                    i18nKey='ssn-or-itin'
                    internalLink='/flow/you-and-your-family/dependents/add-person-basic-info'
                  />
                  <MefAlert
                    type='warning'
                    mefErrorCode='SEIC-F1040-501-02'
                    i18nKey='ssn-or-itin'
                    internalLink='/flow/you-and-your-family/dependents/add-person-basic-info'
                  />
                  <MefAlert
                    type='warning'
                    mefErrorCode='IND-507'
                    i18nKey='ssn-or-itin'
                    internalLink='/flow/you-and-your-family/dependents/add-person-basic-info'
                  />
                  <Tin path='/familyAndHousehold/*/tin' isSensitive={true} />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen route='qualified-dependent-ip-pin-choice'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-choice' />
                  <DFModal i18nKey='/info/ip-pin-choice/what' />
                  <Boolean path='/familyAndHousehold/*/hasIpPin' />
                  <SaveAndOrContinueButton />
                </Screen>
                <Gate condition='/familyAndHousehold/*/hasIpPin'>
                  <Screen route='qualified-dependent-ip-pin-ready'>
                    <TaxReturnAlert
                      i18nKey={`/info/you-and-your-family/dependents/ip-pin-not-ready`}
                      conditions={[
                        `/familyAndHousehold/*/hasIpPin`,
                        { operator: `isFalse`, condition: `/familyAndHousehold/*/flowIpPinReady` },
                        {
                          operator: `isFalseOrIncomplete`,
                          condition: `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                        },
                      ]}
                      type='warning'
                      headingLevel='h2'
                    >
                      <DFAccordion
                        i18nKey='/info/you-and-your-family/dependents/ip-pin-not-ready-explainer'
                        headingLevel='h3'
                        internalLink='/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready'
                      />
                    </TaxReturnAlert>
                    <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-ready' />
                    <HelpLink i18nKey='/info/ip-pin-ready' />
                    <Boolean path='/familyAndHousehold/*/flowIpPinReady' />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen
                    route='qualified-dependent-ip-pin-not-ready'
                    condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/flowIpPinReady` }}
                  >
                    <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-delay' />
                    <HelpLink i18nKey='/info/learn-more-retrieve-ip-pin' />
                    <DFAlert
                      headingLevel='h3'
                      i18nKey={`/info/you-and-your-family/dependents/ip-pin-not-ready-assertion`}
                      type='warning'
                      internalLink='/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready'
                    />
                    <DFModal i18nKey='/info/file-ip-return-without-ip-pin' />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='qualified-dependent-ip-pin-input' condition='/familyAndHousehold/*/flowIpPinReady'>
                    <MefAlert mefErrorCode='IND-996' i18nKey='ip-pin-input' type='warning' />
                    <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-input' />
                    <IpPin path='/familyAndHousehold/*/identityPin' />
                    <SaveAndOrContinueButton />
                  </Screen>
                </Gate>
                <Screen route='qualified-dependent-confirmation'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/qualified-dependent-confirmation' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/qualified-dependent-confirmation' />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>
            </Gate>

            <Gate condition='/familyAndHousehold/*/mayQualifyTPForTaxBenefits'>
              <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/eligibleDependent` }}>
                <Screen
                  route='potential-qp'
                  condition={{
                    operator: `isFalseOrIncomplete`,
                    condition: `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                  }}
                >
                  <Heading i18nKey='/heading/you-and-your-family/dependents/potential-qp' />
                  <DFModal i18nKey='/info/you-and-your-family/dependents/potential-qp' batches={[`cdcc-2`]} />
                  <ConditionalList
                    i18nKey='/info/you-and-your-family/dependents/benefits-list'
                    batches={[`cdcc-2`]}
                    items={mayQualifyTPForTaxBenefitsItems}
                  />
                  <InfoDisplay i18nKey='/info/you-and-your-family/dependents/potential-qp-note' batches={[`cdcc-2`]} />
                  {DependentDoesntQualifyAccordion}
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen route='confirmed-qp' condition='/familyAndHousehold/*/doesQualifyTPForTaxBenefits'>
                  <Heading i18nKey='/heading/you-and-your-family/dependents/confirmed-qp' />
                  {/* QP has at least one completed qualified benefit */}
                  <DFModal
                    i18nKey='/info/you-and-your-family/dependents/confirmed-qp-benefits'
                    condition='/familyAndHousehold/*/doesQualifyTPForTaxBenefits'
                  />
                  <ConditionalList
                    i18nKey='/info/you-and-your-family/dependents/benefits-list'
                    condition='/familyAndHousehold/*/doesQualifyTPForTaxBenefits'
                    batches={[`cdcc-2`]}
                    items={[
                      {
                        itemKey: `hoh`,
                        conditions: [
                          { operator: `isTrueAndComplete`, condition: `/familyAndHousehold/*/personQualifiesTPforHoH` },
                          `/eligibleForHoh`,
                        ],
                      },
                      {
                        itemKey: `qss`,
                        conditions: [
                          { operator: `isTrueAndComplete`, condition: `/familyAndHousehold/*/personQualifiesTPforQss` },
                          `/eligibleForQss`,
                        ],
                      },
                      {
                        itemKey: `cdcc`,
                        conditions: [`/familyAndHousehold/*/personQualifiesTPforCdcc`, `/cdccQualified`],
                      },
                      {
                        itemKey: `eitc`,
                        conditions: [
                          {
                            operator: `isTrueAndComplete`,
                            condition: `/familyAndHousehold/*/personQualifiesTPforEitc`,
                          },
                          `/hasUnclaimedEITCQcsAndQualifiesForEITC`,
                        ],
                      },
                    ]}
                  />
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/dependents/confirmed-possible-qp-benefits'
                    conditions={[
                      `/familyAndHousehold/*/hasIncompletePossibleBenefits`,
                      `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                    ]}
                  />
                  <ConditionalList
                    i18nKey='/info/you-and-your-family/dependents/benefits-list'
                    batches={[`cdcc-2`]}
                    conditions={[
                      `/familyAndHousehold/*/hasIncompletePossibleBenefits`,
                      `/familyAndHousehold/*/doesQualifyTPForTaxBenefits`,
                    ]}
                    items={hasIncompletePossibleBenefitsItems}
                  />
                  <SaveAndOrContinueButton />
                </Screen>
              </Gate>
            </Gate>

            <Gate condition='/isFilingStatusHOH'>
              {/* We will only get to this after a user has set filing status */}
              <Gate condition='/familyAndHousehold/*/isValidSelectedHohQP'>
                {/* The tin for claimed dependents is collected during claim process so we can skip it here */}
                <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/isClaimedDependent` }}>
                  <Screen route='hoh-qp-input-tin'>
                    <Heading i18nKey='/heading/you-and-your-family/dependents/hoh-qp-input-tin' />
                    <Tin path='/familyAndHousehold/*/tin' isSensitive={true} />
                    <SaveAndOrContinueButton />
                  </Screen>
                </Gate>
              </Gate>
            </Gate>
          </SubSubcategory>
          <SubSubcategory route='dependents-qualifying-child-info-unclaimed'>
            <Gate condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/hasTinTypeOfNone` }}>
              <Gate condition='/familyAndHousehold/*/cdccOrEitcNonDependentQualifyingPerson'>
                <Gate condition='/cdccOrEitcQualified'>
                  <Screen route='dependents-eitc-enter-tin'>
                    <TaxReturnAlert
                      type='error'
                      batches={[`eitc-ids-0`]}
                      i18nKey='/info/you-and-your-family/dependents/invalid-ssn'
                      conditions={[
                        `/familyAndHousehold/*/hasInvalidSSN`,
                        `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                      ]}
                    />
                    <TaxReturnAlert
                      type='error'
                      batches={[`eitc-ids-0`]}
                      i18nKey='/info/you-and-your-family/dependents/invalid-itin'
                      conditions={[
                        `/familyAndHousehold/*/hasInvalidITIN`,
                        `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                      ]}
                    />
                    <TaxReturnAlert
                      type='error'
                      batches={[`eitc-ids-0`]}
                      i18nKey='/info/you-and-your-family/dependents/invalid-atin'
                      conditions={[
                        `/familyAndHousehold/*/hasInvalidATIN`,
                        `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                      ]}
                    />
                    <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/enter-tin' />
                    <Tin path='/familyAndHousehold/*/tin' isSensitive={true} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='dependents-eitc-ip-pin-choice'>
                    <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-choice' />
                    <DFModal batches={[`eitc-ids-0`]} i18nKey='/info/ip-pin-choice/what' />
                    <InfoDisplay batches={[`eitc-ids-0`]} i18nKey='/info/ip-pin-choice' />
                    <Boolean path='/familyAndHousehold/*/hasIpPin' />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Gate condition='/familyAndHousehold/*/hasIpPin'>
                    <Screen route='dependents-eitc-ip-pin-ready'>
                      <TaxReturnAlert
                        i18nKey={`/info/you-and-your-family/dependents/ip-pin-not-ready`}
                        conditions={[
                          `/familyAndHousehold/*/hasIpPin`,
                          { operator: `isFalse`, condition: `/familyAndHousehold/*/flowIpPinReady` },
                          `/familyAndHousehold/*/isEITCQCUnclaimedDependent`,
                        ]}
                        type='warning'
                        headingLevel='h2'
                      >
                        <DFAccordion
                          i18nKey='/info/you-and-your-family/dependents/ip-pin-not-ready-explainer'
                          headingLevel='h3'
                          internalLink='/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready'
                        />
                      </TaxReturnAlert>
                      <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-ready' />
                      <HelpLink i18nKey='/info/ip-pin-ready' />
                      <Boolean path='/familyAndHousehold/*/flowIpPinReady' />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Screen
                      route='dependents-eitc-ip-pin-not-ready'
                      condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/flowIpPinReady` }}
                    >
                      <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-delay' />
                      <HelpLink i18nKey='/info/learn-more-retrieve-ip-pin' />
                      <DFAlert
                        headingLevel='h3'
                        i18nKey={`/info/you-and-your-family/dependents/ip-pin-not-ready-assertion`}
                        type='warning'
                        internalLink='/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready'
                      />
                      <DFModal i18nKey='/info/file-ip-return-without-ip-pin' />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Screen route='dependents-eitc-ip-pin-input' condition='/familyAndHousehold/*/flowIpPinReady'>
                      <Heading i18nKey='/heading/you-and-your-family/dependents/ip-pin-input' />
                      <IpPin path='/familyAndHousehold/*/identityPin' />
                      <SaveAndOrContinueButton />
                    </Screen>
                  </Gate>
                </Gate>
              </Gate>
            </Gate>
          </SubSubcategory>
        </Gate>
        <SubSubcategory route='claim-choice'>
          <Gate condition='/familyAndHousehold/*/doesNotQualifyForAnyBenefits'>
            <Screen route='not-qualified'>
              <Heading i18nKey='/heading/you-and-your-family/dependents/not-qualified' />
              <InfoDisplay i18nKey='/info/you-and-your-family/dependents/not-qualified' />
              {DependentDoesntQualifyAccordion}
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
        </SubSubcategory>
      </CollectionLoop>
      <Screen route='exit-person-section' condition='/flowDidNotEnterPeople'>
        <Heading i18nKey='/heading/you-and-your-family/dependents/exit-person-section' />
        <InfoDisplay i18nKey='/info/you-and-your-family/dependents/exit-person-section' />
        <SaveAndOrContinueButton />
      </Screen>
    </Gate>
  </Subcategory>
);
