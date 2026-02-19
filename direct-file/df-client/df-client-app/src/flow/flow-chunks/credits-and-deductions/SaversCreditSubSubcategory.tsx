/* eslint-disable max-len */
import { Gate, Screen, SubSubcategory, Assertion } from '../../flowDeclarations.js';
import {
  Boolean,
  ConditionalAccordion,
  ContextHeading,
  DFModal,
  DFAlert,
  Heading,
  IconDisplay,
  InfoDisplay,
  SaveAndOrContinueButton,
  KnockoutButton,
} from '../../ContentDeclarations.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';

export const SaversCreditDisqualifyingItems: ItemConfig[] = [
  {
    itemKey: `subListSc-noTaxLiability`,
    conditions: [
      { operator: `isFalse`, condition: `/hasTaxLiabilityBeforeNrCredits` },
      `/qualifiedForSaverCreditWoTaxLiability`,
      { operator: `isFalse`, condition: `/qualifiedForSaverCredit` },
    ],
  },
  {
    itemKey: `subListSc-reachedCreditLimit`,
    conditions: [
      `/hasTaxLiabilityBeforeNrCredits`,
      `/qualifiedForSaverCreditWoTaxLiability`,
      { operator: `isFalse`, condition: `/qualifiedForSaverCredit` },
    ],
  },
  {
    itemKey: `subListSc-failsAgiTestforHoh`,
    conditions: [{ operator: `isFalse`, condition: `/isHohAgiEligibleForSaversCredit` }, `/isFilingStatusHOH`],
  },
  {
    itemKey: `subListSc-failsAgiTestforMfj`,
    conditions: [{ operator: `isFalse`, condition: `/isMfjAgiEligibleForSaversCredit` }, `/isFilingStatusMFJ`],
  },
  {
    itemKey: `subListSc-failsAgiTestforOther`,
    conditions: [
      { operator: `isFalse`, condition: `/isOtherAgiEligibleForSaversCredit` },
      { operator: `isFalse`, condition: `/isFilingStatusHOH` },
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    // The single/MFS/QSS/HoH filer did not make eligible contributions.
    itemKey: `subListSc-failsContributionsTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isFalse`, condition: `/didSingleFilerMakeEligibleContributions` },
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    // Only the primary MFJ filer did not make eligible contributions.
    itemKey: `subListSc-failsContributionsTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/didOnlySecondaryMakeEligibleContributions` },
    ],
  },
  {
    // Only the secondary MFJ filer did not make eligible contributions.
    itemKey: `subListSc-failsContributionsTestAsSecondaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/didOnlyPrimaryMakeEligibleContributions` },
    ],
  },
  {
    // Neither MFJ filers made any eligible contributions.
    itemKey: `subListSc-failsContributionsTestAsBothFilers`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/didNeitherFilerMakeEligibleContributions` },
    ],
  },
  {
    // The single/MFS/QSS/HoH filer is under age.
    itemKey: `subListSc-failsAgeTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isSingleFilerUnderAge` },
    ],
  },
  {
    // Only the primary MFJ filer is under age.
    itemKey: `subListSc-failsAgeTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isOnlyPrimaryFilerUnderAge` },
    ],
  },
  {
    // Only the secondary MFJ filer is under age.
    itemKey: `subListSc-failsAgeTestAsSecondaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isOnlySecondaryFilerUnderAge` },
    ],
  },
  {
    // Both MFJ filers are under age.
    itemKey: `subListSc-failsAgeTestAsBothFilers`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/areBothFilersUnderAge` },
    ],
  },
  {
    // The Single/MFS/QSS/HoH filer will be claimed as a dependent.
    itemKey: `subListSc-failsClaimedAsDependentTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/willSingleFilerBeClaimedAsDependent` },
    ],
  },
  {
    // Only the primary MFJ filer will be claimed as a dependent.
    itemKey: `subListSc-failsClaimedAsDependentTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/willOnlyPrimaryFilerBeClaimedAsDependent` },
    ],
  },
  {
    // Only the secondary MFJ filer will be claimed as a dependent.
    itemKey: `subListSc-failsClaimedAsDependentTestAsSecondaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/willOnlySecondaryFilerBeClaimedAsDependent` },
    ],
  },
  {
    // Both MFJ filers will be claimed as dependents.
    // This situation can't really happen.
    itemKey: `subListSc-failsClaimedAsDependentTestAsBothFilers`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/willBothFilersBeClaimedAsDependents` },
    ],
  },
  {
    // The Single/MFS/QSS/HoH filer was a full-time student.
    itemKey: `subListSc-failsFullTimeStudentTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isSingleFilerFullTimeStudent` },
    ],
  },
  {
    // Only the primary MFJ filer was a full-time student.
    itemKey: `subListSc-failsFullTimeStudentTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isOnlyPrimaryFilerFullTimeStudent` },
    ],
  },
  {
    // Only the secondary MFJ filer was a full-time student.
    itemKey: `subListSc-failsFullTimeStudentTestAsSecondaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isOnlySecondaryFilerFullTimeStudent` },
    ],
  },
  {
    // Both MFJ filers were full-time students.
    itemKey: `subListSc-failsFullTimeStudentTestAsBothFilers`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/areBothFilersFullTimeStudents` },
    ],
  },
  {
    // The Single/MFS/QSS/HoH filer had distributions exceeding contributions.
    itemKey: `subListSc-failsDistributionsTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/isSingleFilerFailsDistributionsTest` },
    ],
  },
  {
    // Only the primary MFJ filer had distributions exceeding contributions.
    itemKey: `subListSc-failsDistributionsTestAsPrimaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isFalse`, condition: `/didPrimaryMakeEligibleContributions` },
      { operator: `isTrue`, condition: `/isOnlyPrimaryFilerFailsDistributionsTest` },
    ],
  },
  {
    // Only the secondary MFJ filer had distributions exceeding contributions.
    itemKey: `subListSc-failsDistributionsTestAsSecondaryFilerOnly`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isFalse`, condition: `/didSecondaryMakeEligibleContributions` },
      { operator: `isTrue`, condition: `/isOnlySecondaryFilerFailsDistributionsTest` },
    ],
  },
  {
    // Both MFJ filers had distributions exceeding contributions.
    itemKey: `subListSc-failsDistributionsTestAsBothFilers`,
    conditions: [
      { operator: `isTrue`, condition: `/isAgiEligibleForSaversCredit` },
      { operator: `isTrue`, condition: `/areBothFilersFailDistributionsTest` },
    ],
  },
];

export const SaversCreditSubSubcategory = (
  <Gate condition='/qualifiedForSaverCredit'>
    {/* Uses /qualifiedForSaverCredit instead of /eligibleForSaversCredit because Direct File doesn't support
      contributions not on a W-2, and the only questions in this section are knockouts */}
    <SubSubcategory route='qualified-for-savers-credit' headingLevel='h2' borderStyle='heavy'>
      <Screen route='savers-credit-breather'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/savers'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-breather'
          batches={[`savers-8880-0`]}
          condition='/onlyPrimaryFilerQualifiedForSaversCredit'
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-breather-spouse'
          batches={[`savers-8880-0`]}
          condition='/onlySecondaryFilerQualifiedForSaversCredit'
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-breather-both'
          batches={[`savers-8880-0`]}
          condition='/bothFilersQualifiedForSaversCredit'
        />
        <DFModal i18nKey='/info/credits-and-deductions/credits/savers-explanation' batches={[`savers-8880-0`]} />
        <DFAlert i18nKey={null} headingLevel='h2' type='info'>
          <DFModal i18nKey='/info/credits-and-deductions/credits/savers-not-supported-situations' />
        </DFAlert>
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='savers-self-reported-distribution'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/savers'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-self-reported-distribution'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-self-reported-distribution-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`savers-8880-0`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/savers-self-reported-distribution'
          batches={[`savers-8880-0`]}
        />
        <Boolean path='/hasDistributionsInLastThreeYears' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen
        route='savers-current-year-distributions'
        condition={{ operator: `isFalse`, condition: `/hasDistributionsInLastThreeYears` }}
      >
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/savers'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-current-year-distributions'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-current-year-distributions-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`savers-8880-0`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/savers-current-year-distributions'
          batches={[`savers-8880-0`]}
        />
        <Boolean path='/hasCurrentYearDistributions' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen
        route='savers-self-reported-deferrals-and-contributions'
        condition={{ operator: `isFalse`, condition: `/hasSelfReportedDistributions` }}
      >
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/savers'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-self-reported-contributions'
          condition='/onlyPrimaryFilerEligibleForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-self-reported-contributions-spouse'
          condition='/onlySecondaryFilerEligibleForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-self-reported-contributions-both'
          condition='/bothFilersEligibleForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <DFModal
          i18nKey='/info/credits-and-deductions/credits/savers-self-reported-contributions'
          batches={[`savers-8880-0`]}
        />
        <Boolean
          path='/hasSelfReportedContributions'
          i18nKeySuffixContext='onlyPrimary'
          condition='/onlyPrimaryFilerEligibleForSaversCredit'
        />
        <Boolean
          path='/hasSelfReportedContributions'
          i18nKeySuffixContext='onlySecondary'
          condition='/onlySecondaryFilerEligibleForSaversCredit'
        />
        <Boolean
          path='/hasSelfReportedContributions'
          i18nKeySuffixContext='both'
          condition='/bothFilersEligibleForSaversCredit'
        />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='savers-zero-credit' condition='/meetsOneDollarCombinedCreditLimit'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/savers' />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-eligible-single'
          condition='/singleFilerQualifiedForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-eligible-only-you'
          condition='/onlyPrimaryFilerQualifiedForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-eligible-only-spouse'
          condition='/onlySecondaryFilerQualifiedForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/savers-eligible-both'
          condition='/bothFilersQualifiedForSaversCredit'
          batches={[`savers-8880-0`]}
        />
        <InfoDisplay i18nKey='/info/credits-and-deductions/credits/savers-eligible-zero' batches={[`savers-8880-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <SubSubcategory route='savers-qualified-outcomes' editable={false} headingLevel='h2'>
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.saversPrimaryQualifiedNonMfj'
          conditions={[
            { operator: `isTrue`, condition: `/qualifiedForSaverCredit` },
            { operator: `isFalse`, condition: `/hasSelfReportedDistributionsOrContributions` },
            { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          ]}
        />
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.saversBothQualified'
          conditions={[
            { operator: `isTrue`, condition: `/qualifiedForSaverCredit` },
            { operator: `isFalse`, condition: `/hasSelfReportedDistributionsOrContributions` },
            { operator: `isTrue`, condition: `/isFilingStatusMFJ` },
            { operator: `isTrue`, condition: `/bothFilersQualifiedForSaversCredit` },
          ]}
        />
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.saversPrimaryQualifiedOnlyMfj'
          conditions={[
            { operator: `isTrue`, condition: `/qualifiedForSaverCredit` },
            { operator: `isFalse`, condition: `/hasSelfReportedDistributionsOrContributions` },
            { operator: `isTrue`, condition: `/isFilingStatusMFJ` },
            { operator: `isTrue`, condition: `/onlyPrimaryFilerQualifiedForSaversCredit` },
          ]}
        />
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.saversSpouseQualifiedOnly'
          conditions={[
            { operator: `isTrue`, condition: `/qualifiedForSaverCredit` },
            { operator: `isFalse`, condition: `/hasSelfReportedDistributionsOrContributions` },
            { operator: `isTrue`, condition: `/isFilingStatusMFJ` },
            { operator: `isTrue`, condition: `/onlySecondaryFilerQualifiedForSaversCredit` },
          ]}
        />
        <Screen route='savers-eligible' condition='/flowShowSaversEligibleScreen'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/savers' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/savers-eligible-single'
            condition='/singleFilerQualifiedForSaversCredit'
            batches={[`savers-8880-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/savers-eligible-only-you'
            condition='/onlyPrimaryFilerQualifiedForSaversCredit'
            batches={[`savers-8880-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/savers-eligible-only-spouse'
            condition='/onlySecondaryFilerQualifiedForSaversCredit'
            batches={[`savers-8880-0`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/savers-eligible-both'
            condition='/bothFilersQualifiedForSaversCredit'
            batches={[`savers-8880-0`]}
          />
          <ConditionalAccordion
            i18nKey='/info/credits-and-deductions/credits/savers-why-i-not-qual'
            batches={[`savers-8880-0`]}
            condition='/onlySecondaryFilerQualifiedForSaversCredit'
            items={[
              {
                itemKey: `subList`,
              },
              {
                itemKey: `subList-under18`,
                conditions: [`/primaryFiler/isUnder18`],
              },
              {
                itemKey: `subList-dependent`,
                conditions: [`/primaryFiler/willBeClaimed`],
              },
              {
                itemKey: `subList-student`,
                conditions: [`/primaryFiler/isStudent`],
              },
              {
                itemKey: `subList-contributions`,
                conditions: [{ operator: `isFalse`, condition: `/didPrimaryMakeEligibleContributions` }],
              },
              {
                itemKey: `subList-failsDistributionTest`,
                conditions: [{ operator: `isTrue`, condition: `/isOnlyPrimaryFilerFailsDistributionsTest` }],
              },
            ]}
          />
          <ConditionalAccordion
            i18nKey='/info/credits-and-deductions/credits/savers-why-spouse-not-qual'
            batches={[`savers-8880-0`]}
            condition='/onlyPrimaryFilerQualifiedForSaversCredit'
            items={[
              {
                itemKey: `subList`,
              },
              {
                itemKey: `subList-under18`,
                conditions: [`/secondaryFiler/isUnder18`],
              },
              {
                itemKey: `subList-dependent`,
                conditions: [`/secondaryFiler/willBeClaimed`],
              },
              {
                itemKey: `subList-student`,
                conditions: [`/secondaryFiler/isStudent`],
              },
              {
                itemKey: `subList-contributions`,
                conditions: [{ operator: `isFalse`, condition: `/didSecondaryMakeEligibleContributions` }],
              },
              {
                itemKey: `subList-failsDistributionTest`,
                conditions: [{ operator: `isTrue`, condition: `/isOnlySecondaryFilerFailsDistributionsTest` }],
              },
            ]}
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/savers-eligible' batches={[`savers-8880-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Screen
        route='savers-self-reported-distributions-ko'
        condition='/hasDistributionsInLastThreeYears'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/savers-self-reported-distributions-ko' batches={[`savers-8880-0`]} />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/savers-self-reported-distributions-ko'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`savers-8880-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/savers-self-reported-distributions-ko-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`savers-8880-0`]}
        />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      <Screen route='savers-current-year-distributions-ko' condition='/hasCurrentYearDistributions' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/savers-current-year-distributions-ko' batches={[`savers-8880-0`]} />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/savers-current-year-distributions-ko'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`savers-8880-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/savers-current-year-distributions-ko-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`savers-8880-0`]}
        />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      <Screen route='savers-self-reported-contributions-ko' condition='/hasSelfReportedContributions' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/savers-self-reported-contributions-ko' batches={[`savers-8880-0`]} />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/savers-self-reported-contributions-ko'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          batches={[`savers-8880-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/savers-self-reported-contributions-ko-mfj'
          condition='/isFilingStatusMFJ'
          batches={[`savers-8880-0`]}
        />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>
  </Gate>
);
