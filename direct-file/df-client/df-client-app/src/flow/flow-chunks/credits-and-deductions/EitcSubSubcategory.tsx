/* eslint-disable max-len */
import { Assertion, CollectionLoop, Gate, Screen, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ContextHeading,
  DatePicker,
  DFModal,
  DFAlert,
  Heading,
  HelpLink,
  IconDisplay,
  IpPin,
  SaveAndOrContinueButton,
  SaveAndOrContinueAndSetFactButton,
  InfoDisplay,
  Enum,
  ConditionalList,
  InternalLink,
  MefAlert,
  Tin,
  TaxReturnAlert,
} from '../../ContentDeclarations.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';

export const EitcDisqualifyingItems: ItemConfig[] = [
  {
    itemKey: `subListEitc-agiLimitsSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/belowEitcAgiLimit` },
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    itemKey: `subListEitc-agiLimitsMfj`,
    conditions: [{ operator: `isFalse`, condition: `/belowEitcAgiLimit` }, `/isFilingStatusMFJ`],
  },
  {
    itemKey: `subListEitc-validSsnSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/filersHaveValidSSNsForEitc` },
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    itemKey: `subListEitc-validSsnMfj`,
    conditions: [{ operator: `isFalse`, condition: `/filersHaveValidSSNsForEitc` }, `/isFilingStatusMFJ`],
  },
  {
    itemKey: `subListEitc-specialSpouseRule`,
    conditions: [`/noEitcDueToSeparateFilingFromSpouse`],
  },
  // We have the following key twice once for the MFJ case and one for the non MFJ case since they require
  // different checks but show the same content
  {
    itemKey: `subListEitc-validCitizenStatusSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
      { operator: `isFalse`, condition: `/filersUSCitizenOrRAAllYear` },
      { operator: `isFalse`, condition: `/primaryFilerIsCitizenOrRAAllYear` },
    ],
  },
  {
    itemKey: `subListEitc-validCitizenStatusSelf`,
    conditions: [
      `/isFilingStatusMFJ`,
      { operator: `isFalse`, condition: `/filersUSCitizenOrRAAllYear` },
      { operator: `isFalse`, condition: `/primaryFilerIsCitizenOrRAAllYear` },
      `/secondaryFilerIsCitizenOrRAAllYear`,
    ],
  },
  {
    itemKey: `subListEitc-validCitizenStatusSpouse`,
    conditions: [
      { operator: `isFalse`, condition: `/filersUSCitizenOrRAAllYear` },
      `/isFilingStatusMFJ`,
      `/primaryFilerIsCitizenOrRAAllYear`,
      { operator: `isFalse`, condition: `/secondaryFilerIsCitizenOrRAAllYear` },
    ],
  },
  {
    itemKey: `subListEitc-validCitizenStatusBoth`,
    conditions: [
      { operator: `isFalse`, condition: `/filersUSCitizenOrRAAllYear` },
      `/isFilingStatusMFJ`,
      { operator: `isFalse`, condition: `/primaryFilerIsCitizenOrRAAllYear` },
      { operator: `isFalse`, condition: `/secondaryFilerIsCitizenOrRAAllYear` },
    ],
  },
  {
    itemKey: `subListEitc-noEarnedIncome`,
    conditions: [{ operator: `isFalse`, condition: `/hasEarnedIncome` }],
  },
  {
    itemKey: `subListEitc-earnedIncomeLimitSelf`,
    conditions: [
      // Prefer the AGI messaging over the earned income messaging, if both apply
      { operator: `isTrueOrIncomplete`, condition: `/belowEitcAgiLimit` },
      { operator: `isFalse`, condition: `/belowEitcEarnedIncomeLimit` },
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    itemKey: `subListEitc-earnedIncomeLimitMfj`,
    conditions: [
      // Prefer the AGI messaging over the earned income messaging, if both apply
      { operator: `isTrueOrIncomplete`, condition: `/belowEitcAgiLimit` },
      { operator: `isFalse`, condition: `/belowEitcEarnedIncomeLimit` },
      `/isFilingStatusMFJ`,
    ],
  },
  {
    // atLeastOneChildCanBeQCOfAnother can only happen if they pass the RelAgeResJointTests so we don't need to directly test it
    // And thus this reason takes priority over `noQCBecauseFailedRelAgeResJointTests`
    itemKey: `subListEitc-noQCBecauseMultipleClaimants`,
    conditions: [
      `/hasZeroEitcQualifyingChildren`,
      `/isFilingStatusMFS`,
      `/maybeEligibleForEitcBase`,
      `/atLeastOneChildCanBeQCOfAnother`,
    ],
  },
  {
    itemKey: `subListEitc-noQCBecauseFailedRelAgeResJointTests`,
    conditions: [
      `/hasZeroEitcQualifyingChildren`,
      `/isFilingStatusMFS`,
      { operator: `isFalse`, condition: `/atLeastOneChildCanBeQCOfAnother` },
      `/maybeEligibleForEitcBase`,
      `/atLeastOneChildFailsEitcRelAgeResJointTest`,
    ],
  },
  {
    itemKey: `subListEitc-filerAgeTestSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
      { operator: `isFalse`, condition: `/isFilingStatusMFS` },
      `/hasZeroEitcQualifyingChildren`,
      `/maybeEligibleForEitcBase`,
      { operator: `isFalse`, condition: `/filersAgeTestForEitcWithNoQC` },
    ],
  },
  {
    itemKey: `subListEitc-filerAgeTestMfj`,
    // checking /isFilingStatusMFJ implicitly checks isFalse:/isFilingStatusMFS
    conditions: [
      `/isFilingStatusMFJ`,
      `/hasZeroEitcQualifyingChildren`,
      `/maybeEligibleForEitcBase`,
      { operator: `isFalse`, condition: `/filersAgeTestForEitcWithNoQC` },
    ],
  },
  {
    itemKey: `subListEitc-potentialDependent`,
    conditions: [
      { operator: `isFalse`, condition: `/isFilingStatusMFS` },
      `/hasZeroEitcQualifyingChildren`,
      `/maybeEligibleForEitcBase`,
      `/treatFilersAsDependents`,
    ],
  },

  {
    itemKey: `subListEitc-qcOfAnother`,
    conditions: [{ operator: `isFalse`, condition: `/eitcQcTest` }, `/maybeEligibleForEitc`],
  },
  {
    itemKey: `subListEitc-improperClaimsSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/eitcNoticeTest` },
      `/maybeEligibleForEitc`,
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    itemKey: `subListEitc-improperClaimsMfj`,
    conditions: [{ operator: `isFalse`, condition: `/eitcNoticeTest` }, `/maybeEligibleForEitc`, `/isFilingStatusMFJ`],
  },
];
export const EitcSubSubcategory = (
  <Gate condition='/maybeEligibleForEitc'>
    <SubSubcategory route='qualifying-child' headingLevel='h3' borderStyle='heavy'>
      <Screen route='eitc-breather'>
        <ContextHeading i18nKey='/heading/credits-and-deductions/credits/eitc' />
        <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-breather' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='eitc-taxpayer-qc-of-another'>
        <ContextHeading i18nKey='/heading/credits-and-deductions/credits/eitc' />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/eitc-taxpayer-qc-of-another-mfj'
          condition='/isFilingStatusMFJ'
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/eitc-taxpayer-qc-of-another'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-taxpayer-qc-of-another' />
        <Boolean path='/eitcQcOfAnother' />
        <SaveAndOrContinueButton />
      </Screen>
      <Gate condition='/eitcQcOfAnother'>
        <Screen route='eitc-qc-of-another-claimer-filing-requirement'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-qc-of-another-claimer-filing-requirement-mfj'
            condition='/isFilingStatusMFJ'
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-qc-of-another-claimer-filing-requirement'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <DFModal i18nKey='/info/credits-and-deductions/credits/required-to-file-snack' />
          <Boolean path='/eitcQcOfAnotherRequiredToFile' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='eitc-qc-of-another-claimer-filing'
          condition={{ operator: `isFalse`, condition: `/eitcQcOfAnotherRequiredToFile` }}
        >
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-qc-of-another-claimer-filing' />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/eitc-qc-of-another-claimer-filing' />
          <Boolean path='/eitcQcOfAnotherIsFiling' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='eitc-qc-of-another-claimers-return' condition='/eitcQcOfAnotherIsFiling'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-qc-of-another-claimers-return' />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/eitc-qc-of-another-claimers-return' />
          <Boolean path='/eitcQcOfAnotherIsFilingRefundOnly' />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
    <Screen route='eitc-qc-of-another-summary' condition='/flowShowNotEitcQc'>
      <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
      <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-qc-of-another-summary' />
      <InfoDisplay i18nKey='/info/credits-and-deductions/credits/eitc-qc-of-another-summary' />
      <SaveAndOrContinueButton />
    </Screen>

    <Gate condition='/eitcQcTest'>
      <SubSubcategory route='eitc-reduced-disallowed' headingLevel='h3'>
        <Screen route='eitc-improper-claims'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-mfj'
            condition='/isFilingStatusMFJ'
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/eitc-improper-claims' />
          <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-improper-claims' />
          <Boolean
            path='/eitcHadImproperClaims'
            i18nKeySuffixContext='self'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Boolean path='/eitcHadImproperClaims' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='eitc-improper-claims-no-qc' condition='/hasImproperClaimsAndNoQc'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-no-qc'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-no-qc-mfj'
            condition='/isFilingStatusMFJ'
          />
          <Boolean path='/eitcImproperClaimsDueToQc' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='eitc-improper-claims-taken-again-since-disallowal'
          condition='/flowHasImproperClaimsAndAskAboutLastClaimedCredit'
        >
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-taken-again-since-disallowal'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-taken-again-since-disallowal-mfj'
            condition='/isFilingStatusMFJ'
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/improper-claims-taken-again-since-disallowal' />
          <Boolean
            path='/eitcImproperClaimTakenSinceDisallowal'
            i18nKeySuffixContext='self'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Boolean
            path='/eitcImproperClaimTakenSinceDisallowal'
            i18nKeySuffixContext='mfj'
            condition='/isFilingStatusMFJ'
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='eitc-improper-claims-notice'
          condition={{ operator: `isFalse`, condition: `/eitcImproperClaimTakenSinceDisallowal` }}
        >
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-notice'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-notice-mfj'
            condition='/isFilingStatusMFJ'
          />
          <DFModal i18nKey='/info/credits-and-deductions/credits/improper-claims-notice' />
          <Enum
            path='/eitcReceivedImproperClaimsNotice'
            i18nKeySuffixContext='self'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Enum path='/eitcReceivedImproperClaimsNotice' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='eitc-improper-claims-waiting-period' condition='/hadEitcImproperClaimsBan'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-waiting-period'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-improper-claims-waiting-period-mfj'
            condition='/isFilingStatusMFJ'
          />
          <Boolean path='/eitcImproperClaimsNoticeExpired' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </Gate>
    <SubSubcategory route='qualifying-children' headingLevel='h3'>
      <Assertion
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.noChildrenButqualifiesForEITC'
        type={`info`}
        conditions={[`/hasZeroEitcQualifyingChildren`, `/eitcQualified`]}
      />
      <Assertion
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.numberofChildrenQualifyingForEITC'
        type={`info`}
        conditions={[
          `/eitcQualified`,
          `/hasMoreThanZeroEitcQualifyingChildren`,
          { operator: `isFalse`, condition: `/eitcQCsNeedAdditionalInfo` },
        ]}
      />
      <Assertion
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.numberofChildrenQualifyingForEITCNeedAdditionalInfo'
        type={`info`}
        conditions={[`/eitcQualified`, `/hasMoreThanZeroEitcQualifyingChildren`, `/eitcQCsNeedAdditionalInfo`]}
      />
    </SubSubcategory>
    <SubSubcategory route='eitc-not-qualified-outcome' editable={false} headingLevel='h3'>
      <Assertion
        type='info'
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.eitcNotQualified'
        conditions={[{ operator: `isFalseAndComplete`, condition: `/eitcQualified` }, `/maybeEligibleForEitc`]}
      />
      <Screen route='eitc-not-qualified' condition={{ operator: `isFalse`, condition: `/eitcQualified` }}>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
        <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-not-qualified' />
        <InfoDisplay i18nKey='/info/credits-and-deductions/credits/eitc-not-qualified-reasons' />
        <ConditionalList
          i18nKey='/info/credits-and-deductions/credits/eitc-not-qualified-reasons'
          items={[
            { itemKey: `qc`, conditions: [{ operator: `isFalse`, condition: `/eitcQcTest` }] },
            {
              itemKey: `improper-claims`,
              conditions: [
                { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/eitcNoticeTest` },
              ],
            },
            {
              itemKey: `improper-claims-mfj`,
              conditions: [`/isFilingStatusMFJ`, { operator: `isFalse`, condition: `/eitcNoticeTest` }],
            },
          ]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <Gate condition='/eitcQualified'>
      <SubSubcategory route='eitc-qualified-outcomes' editable={false} headingLevel='h3'>
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.eitcQualifiedManyChildren'
          conditions={[`/hasManyEitcQualifyingChildren`, `/eitcQualified`]}
        />
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.eitcQualifiedOneChild'
          conditions={[
            { operator: `isFalse`, condition: `/hasManyEitcQualifyingChildren` },
            { operator: `isFalse`, condition: `/hasZeroEitcQualifyingChildren` },
            `/eitcQualified`,
          ]}
        />
        <Assertion
          type='success'
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.eitcQualifiedNoChildren'
          conditions={[`/hasZeroEitcQualifyingChildren`, `/eitcQualified`]}
        />
        <Screen route='eitc-qualified'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-qualified' />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/eitc-qualified-with-qcs'
            condition={`/maybeEligibleForEitcWithQc`}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/eitc-qualified-no-qcs'
            condition={{ operator: `isFalse`, condition: `/maybeEligibleForEitcWithQc` }}
          />
          <ConditionalList
            i18nKey='/info/credits-and-deductions/credits/eitc-qualified-qcs'
            items={[
              {
                itemKey: `primary`,
                collection: `/eitcDependentsCollection`,
                conditions: [`/maybeEligibleForEitcWithQc`],
              },
            ]}
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/eitc-qualified-explain' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>

      {/* If we need to fill out form 8862 and any QCs are deceased, we need to ask month and day of death */}
      <CollectionLoop
        collection='/deceasedEitcEligibleQcCollection'
        loopName='/deceasedEitcEligibleQcCollection'
        autoIterate={true}
        collectionItemCompletedCondition='/hasNeededDeceasedInfo'
      >
        <SubSubcategory route='qualifying-child-info' headingLevel='h3'>
          <Screen route='eitc-8862-date-of-death' condition='/form8862RequiredAndHasQualifyingChildren'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
            <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-8862-date-of-death' />
            <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-8862-date-of-death' />
            <DatePicker path='/familyAndHousehold/*/dateOfDeath' lockYearTo='/taxYear' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </CollectionLoop>
      {/* If we are claiming QCs who are not dependents, then we need to collect TIN/PIN info if they have those*/}
      <Gate condition='/hasUnclaimedEITCQcsWithTINs'>
        {/* But first, a breather */}
        <Screen route='eitc-additional-information-for-nd' condition='/hasAnyUnclaimedEITCQCs'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-additional-information-for-nd-single'
            condition={{ operator: `isFalseOrIncomplete`, condition: `/hasMoreThanOneUnclaimedEITCQC` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/eitc-additional-information-for-nd-several'
            condition='/hasMoreThanOneUnclaimedEITCQC'
          />
          <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-additional-information-for-nd' />
          <SaveAndOrContinueButton />
        </Screen>
        <CollectionLoop
          collection='/unclaimedEITCQcsWithTINsCollection'
          loopName='/unclaimedEITCQcsWithTINsCollection'
          autoIterate={true}
          collectionItemCompletedCondition='/hasNoUnclaimedEITCQcsCollectionWhoNeedTINs'
        >
          <SubSubcategory route='qualifying-child-info' headingLevel='h3'>
            <Screen route='eitc-enter-tin'>
              <MefAlert
                mefErrorCode='SEIC-F1040-535-04'
                i18nKey='tin'
                type='warning'
                internalLink='/flow/you-and-your-family/dependents/add-person-basic-info'
                condition='/hasUnclaimedEITCQcsWithTINs'
              />
              <ContextHeading
                displayOnlyOn='edit'
                batches={[`eitc-ids-0`]}
                i18nKey='/heading/credits-and-deductions/credits/eitc'
              />
              <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/enter-tin' />
              <Tin path='/familyAndHousehold/*/tin' isSensitive={true} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='eitc-ip-pin-choice'>
              <ContextHeading
                displayOnlyOn='edit'
                batches={[`eitc-ids-0`]}
                i18nKey='/heading/credits-and-deductions/credits/eitc'
              />
              <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-choice' />
              <DFModal batches={[`eitc-ids-0`]} i18nKey='/info/ip-pin-choice/what' />
              <InfoDisplay batches={[`eitc-ids-0`]} i18nKey='/info/ip-pin-choice' />
              <Boolean path='/familyAndHousehold/*/hasIpPin' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <Gate condition='/familyAndHousehold/*/hasIpPin'>
            <SubSubcategory route='qualifying-child-info' headingLevel='h3'>
              <Screen route='eitc-ip-pin-ready'>
                <TaxReturnAlert
                  i18nKey={`/info/ip-pin-not-ready`}
                  conditions={[
                    `/familyAndHousehold/*/hasIpPin`,
                    { operator: `isFalse`, condition: `/familyAndHousehold/*/flowIpPinReady` },
                  ]}
                  type='warning'
                />
                <ContextHeading
                  displayOnlyOn='edit'
                  batches={[`eitc-ids-0`]}
                  i18nKey='/heading/credits-and-deductions/credits/eitc'
                />
                <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-ready' />
                <InfoDisplay batches={[`eitc-ids-0`]} i18nKey='/info/credits-and-deductions/credits/ip-pin-ready' />
                <Boolean path='/familyAndHousehold/*/flowIpPinReady' />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
            <Screen
              route='eitc-ip-pin-not-ready'
              condition={{ operator: `isFalse`, condition: `/familyAndHousehold/*/flowIpPinReady` }}
            >
              <ContextHeading
                displayOnlyOn='edit'
                batches={[`eitc-ids-0`]}
                i18nKey='/heading/credits-and-deductions/credits/eitc'
              />
              <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-delay' />
              <HelpLink batches={[`eitc-ids-0`]} i18nKey='/info/learn-more-retrieve-ip-pin' />
              <DFAlert batches={[`eitc-ids-0`]} i18nKey='/info/ip-pin-not-ready' headingLevel='h3' type='warning' />
              <DFModal batches={[`eitc-ids-0`]} i18nKey='/info/file-ip-return-without-ip-pin' />
              <SaveAndOrContinueButton />
            </Screen>
            <SubSubcategory route='qualifying-child-info' headingLevel='h3'>
              <Screen route='eitc-ip-pin-input' condition='/familyAndHousehold/*/flowIpPinReady'>
                <ContextHeading
                  displayOnlyOn='edit'
                  batches={[`eitc-ids-0`]}
                  i18nKey='/heading/credits-and-deductions/credits/eitc'
                />
                <Heading batches={[`eitc-ids-0`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-input' />
                <IpPin path='/familyAndHousehold/*/identityPin' />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
          </Gate>
          <Screen route='eitc-qc-confirmation'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
            <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-qc-confirmation' />
            <SaveAndOrContinueButton />
          </Screen>
        </CollectionLoop>
      </Gate>
      <Gate condition='/someFilerHasCombatPay'>
        <SubSubcategory route='eitc-combat-pay-earned-income' headingLevel='h3'>
          <Screen route='eitc-primary-filer-combat-pay-recommendation' condition='/onlyPrimaryFilerHasCombatPay'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
            <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-combat-pay-recommendation/primary-filer-has-combat-pay' />
            <DFAlert
              condition='/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay'
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-primary-filer-combat-pay-recommend-to-include' />
            </DFAlert>
            <SaveAndOrContinueAndSetFactButton
              condition='/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay'
              i18nKey='button.continueIncludePrimaryFilerCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition='/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay'
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
            <DFAlert
              i18nKey={null}
              condition={{
                operator: `isFalse`,
                condition: `/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay`,
              }}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-primary-filer-combat-pay-recommend-to-exclude' />
            </DFAlert>
            <SaveAndOrContinueAndSetFactButton
              condition={{
                operator: `isFalse`,
                condition: `/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay`,
              }}
              i18nKey='button.continueExcludePrimaryFilerCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition={{
                operator: `isFalse`,
                condition: `/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay`,
              }}
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
          </Screen>
          <Screen route='eitc-mfj-spouse-combat-pay-recommendation' condition='/onlyMFJSpouseHasCombatPay'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
            <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-combat-pay-recommendation/mfj-spouse-has-combat-pay' />
            <DFAlert
              condition='/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay'
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-secondary-filer-combat-pay-recommend-to-include' />
            </DFAlert>
            <SaveAndOrContinueAndSetFactButton
              condition='/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay'
              i18nKey='button.continueIncludeMFJSpouseCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition='/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay'
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
            <DFAlert
              condition={{
                operator: `isFalse`,
                condition: `/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay`,
              }}
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-secondary-filer-combat-pay-recommend-to-exclude' />
            </DFAlert>
            <SaveAndOrContinueAndSetFactButton
              condition={{
                operator: `isFalse`,
                condition: `/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay`,
              }}
              i18nKey='button.continueExcludeMFJSpouseCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition={{
                operator: `isFalse`,
                condition: `/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay`,
              }}
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
          </Screen>
          <Screen route='eitc-both-filers-combat-pay-recommendation' condition='/bothFilersHasCombatPay'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
            <Heading i18nKey='/heading/credits-and-deductions/credits/eitc-combat-pay-recommendation/both-filers-has-combat-pay' />
            {/* Recommend to use both combat pay*/}
            <DFAlert condition='/isCombatPayRecommendationUseBoth' i18nKey={null} headingLevel='h2' type='success'>
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-include' />
            </DFAlert>
            <SaveAndOrContinueAndSetFactButton
              condition='/isCombatPayRecommendationUseBoth'
              i18nKey='button.continueIncludeBothFilersCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition='/isCombatPayRecommendationUseBoth'
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
            {/* Recommend to use Primary not Spouse combat pay*/}
            <DFAlert
              condition='/isCombatPayRecommendationUsePrimaryNotSpouse'
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-include-primary-exclude-secondary' />
            </DFAlert>

            <SaveAndOrContinueAndSetFactButton
              condition='/isCombatPayRecommendationUsePrimaryNotSpouse'
              i18nKey='button.continueIncludePrimaryExcludeSecondaryFilersCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition='/isCombatPayRecommendationUsePrimaryNotSpouse'
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
            {/* Recommend to Spouse not Primary combat pay*/}
            <DFAlert
              condition='/isCombatPayRecommendationUseSpouseNotPrimary'
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-include-secondary-exclude-primary' />
            </DFAlert>
            <SaveAndOrContinueAndSetFactButton
              condition='/isCombatPayRecommendationUseSpouseNotPrimary'
              i18nKey='button.continueIncludeSecondaryExcludePrimaryFilersCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition='/isCombatPayRecommendationUseSpouseNotPrimary'
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
            {/* Recommend to use none combat pay*/}
            <DFAlert condition='/isCombatPayRecommendationNoCombatPay' i18nKey={null} headingLevel='h2' type='success'>
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-exclude' />
            </DFAlert>

            <SaveAndOrContinueAndSetFactButton
              condition='/isCombatPayRecommendationNoCombatPay'
              i18nKey='button.continueExcludeBothFilersCombatPay'
              sourcePath='/combatPayRecommendation'
              destinationPath='/combatPayElection'
            />
            <InternalLink
              condition='/isCombatPayRecommendationNoCombatPay'
              i18nKey='/info/credits-and-deductions/credits/combat-pay-recommendation-choose-another-option'
              route='/flow/credits-and-deductions/credits/eitc-combat-pay-change'
            />
          </Screen>
          <Screen route='eitc-combat-pay-change' routeAutomatically={false}>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/eitc' />
            {/* only primary has combat pay */}
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/eitc-combat-pay-change/primary-filer-has-combat-pay'
              condition='/onlyPrimaryFilerHasCombatPay'
            />
            <DFAlert
              conditions={[
                `/onlyPrimaryFilerHasCombatPay`,
                `/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay`,
              ]}
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-primary-filer-combat-pay-recommend-to-exclude' />
            </DFAlert>
            <DFAlert
              conditions={[
                `/onlyPrimaryFilerHasCombatPay`,
                { operator: `isFalse`, condition: `/isEITCAmountWithPrimaryCombatPayGreaterThanWithoutCombatPay` },
              ]}
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-primary-filer-combat-pay-recommend-to-exclude' />
            </DFAlert>
            {/* only spouse has combat pay */}
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/eitc-combat-pay-change/mfj-spouse-has-combat-pay'
              condition='/onlyMFJSpouseHasCombatPay'
            />
            <DFAlert
              conditions={[
                `/onlyMFJSpouseHasCombatPay`,
                `/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay`,
              ]}
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-secondary-filer-combat-pay-recommend-to-include' />
            </DFAlert>
            <DFAlert
              conditions={[
                `/onlyMFJSpouseHasCombatPay`,
                {
                  operator: `isFalse`,
                  condition: `/isEITCAmountWithSecondaryCombatPayGreaterThanWithoutCombatPay`,
                },
              ]}
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-secondary-filer-combat-pay-recommend-to-exclude' />
            </DFAlert>
            {/* both primary and spouse have combat pay */}
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/eitc-combat-pay-change/both-filers-has-combat-pay'
              condition='/bothFilersHasCombatPay'
            />
            <DFAlert condition='/isCombatPayRecommendationUseBoth' i18nKey={null} headingLevel='h2' type='success'>
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-include' />
            </DFAlert>
            <DFAlert
              condition='/isCombatPayRecommendationUsePrimaryNotSpouse'
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-include-primary-exclude-secondary' />
            </DFAlert>
            <DFAlert
              condition='/isCombatPayRecommendationUseSpouseNotPrimary'
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-include-secondary-exclude-primary' />
            </DFAlert>
            <DFAlert
              conditions={[`/bothFilersHasCombatPay`, `/isCombatPayRecommendationNoCombatPay`]}
              i18nKey={null}
              headingLevel='h2'
              type='success'
            >
              <DFModal i18nKey='/info/credits-and-deductions/credits/eitc-both-filers-combat-pay-recommendation-to-exclude' />
            </DFAlert>
            <Enum path='/combatPayElection' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Gate>
    </Gate>
  </Gate>
);
