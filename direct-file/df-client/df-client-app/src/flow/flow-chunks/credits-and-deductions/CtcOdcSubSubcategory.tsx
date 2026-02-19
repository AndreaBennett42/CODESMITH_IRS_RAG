/* eslint-disable max-len */
import { Assertion, Gate, Screen, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ContextHeading,
  DFModal,
  Heading,
  IconDisplay,
  SaveAndOrContinueButton,
  InfoDisplay,
  Enum,
  ConditionalList,
} from '../../ContentDeclarations.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';

export const CtcDisqualifyingItems: ItemConfig[] = [
  {
    itemKey: `subListCtc-failedDependencyTest`,
    conditions: [`/treatFilersAsDependents`],
  },
  {
    itemKey: `subListCtc-noCtcQualifyingChildren`,
    conditions: [
      { operator: `isFalse`, condition: `/maybeEligibleForCtc` },
      { operator: `isFalse`, condition: `/treatFilersAsDependents` },
    ],
  },
  {
    itemKey: `subListCtc-improperClaimsSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/treatFilersAsDependents` },
      `/ctcOdcDisqualifiedForImproperClaims`,
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    itemKey: `subListCtc-improperClaimsMfj`,
    conditions: [
      { operator: `isFalse`, condition: `/treatFilersAsDependents` },
      `/ctcOdcDisqualifiedForImproperClaims`,
      `/isFilingStatusMFJ`,
    ],
  },
];
export const OdcDisqualifyingItems: ItemConfig[] = [
  {
    itemKey: `subListOdc-failedDependencyTest`,
    conditions: [`/treatFilersAsDependents`],
  },
  {
    itemKey: `subListOdc-noOdcQualifyingChildren`,
    conditions: [
      { operator: `isFalse`, condition: `/maybeEligibleForOdc` },
      { operator: `isFalse`, condition: `/treatFilersAsDependents` },
    ],
  },
  {
    itemKey: `subListOdc-improperClaimsSelf`,
    conditions: [
      { operator: `isFalse`, condition: `/treatFilersAsDependents` },
      `/ctcOdcDisqualifiedForImproperClaims`,
      { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
    ],
  },
  {
    itemKey: `subListOdc-improperClaimsMfj`,
    conditions: [
      { operator: `isFalse`, condition: `/treatFilersAsDependents` },
      `/ctcOdcDisqualifiedForImproperClaims`,
      `/isFilingStatusMFJ`,
    ],
  },
];

export const CtcOdcSubSubcategory = (
  <Gate condition='/flowTrue'>
    <SubSubcategory route='ctc-odc' headingLevel='h2' borderStyle='heavy'>
      <Gate condition='/maybeEligibleForCtc'>
        <Screen route='ctc-intro'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/ctc-intro' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='ctc-improper-claims'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/received-improper-claims'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-mfj'
            condition='/isFilingStatusMFJ'
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/potential-credits-improperly-claimed' />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/improper-claims' />
          <DFModal i18nKey='/info/credits-and-deductions/credits/improper-claims-ctc' />
          <Boolean
            path='/receivedImproperClaims'
            i18nKeySuffixContext='self'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Boolean path='/receivedImproperClaims' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/receivedImproperClaims'>
          <Screen route='ctc-improper-claims-taken-again-since-disallowal'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-taken-again'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-taken-again-mfj'
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/potential-credits-improperly-claimed' />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/improper-claims-taken-again-since-disallowal' />
            <Boolean
              path='/hasFiledCtcOdcSinceNoticeExpired'
              i18nKeySuffixContext='self'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Boolean
              path='/hasFiledCtcOdcSinceNoticeExpired'
              i18nKeySuffixContext='mfj'
              condition='/isFilingStatusMFJ'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='ctc-improper-claims-notice'
            condition={{ operator: `isFalse`, condition: `/hasFiledCtcOdcSinceNoticeExpired` }}
          >
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice-mfj'
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/potential-credits-improperly-claimed' />
            <DFModal i18nKey='/info/credits-and-deductions/credits/improper-claims-notice' />
            <Enum
              path='/receivedImproperClaimsNotice'
              i18nKeySuffixContext='self'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Enum path='/receivedImproperClaimsNotice' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='ctc-improper-claims-waiting-period' condition='/hadCtcOdcImproperClaimsBan'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice-expired-mfj'
              condition='/isFilingStatusMFJ'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice-expired'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Boolean path='/improperClaimsNoticeExpired' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='ctc-odc-outcome-no' headingLevel='h3' editable={false}>
      <Gate condition='/maybeEligibleForCtc'>
        <Assertion
          type={`info`}
          i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcOdcNotQualified'
          conditions={[
            { operator: `isTrue`, condition: `/maybeEligibleForCtc` },
            { operator: `isFalseAndComplete`, condition: `/ctcQualified` },
            { operator: `isFalseAndComplete`, condition: `/odcQualified` },
          ]}
        />
        <Screen route='ctc-not-qualified' condition='/ctcHasImproperClaimsAndMaybeHasAdditionalCredits'>
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/improper-claims-not-qualified'
            batches={[`information-architecture-0`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/improper-claims-not-qualified-explanation'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            batches={[`information-architecture-0`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/improper-claims-not-qualified-explanation-mfj'
            condition='/isFilingStatusMFJ'
            batches={[`information-architecture-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='ctc-only-outcome-yes' headingLevel='h3' editable={false}>
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcQualifiedOne'
        conditions={[
          `/ctcQualified`,
          { operator: `isFalseAndComplete`, condition: `/odcQualified` },
          `/exactlyOneCtcEligibleDependent`,
        ]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcQualifiedMore'
        conditions={[
          `/ctcQualified`,
          { operator: `isFalseAndComplete`, condition: `/odcQualified` },
          `/moreThanOneCtcEligibleDependent`,
        ]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcOneAndOdcOneQualified'
        conditions={[
          `/ctcQualified`,
          `/odcQualified`,
          `/exactlyOneCtcEligibleDependent`,
          `/exactlyOneOdcEligibleDependent`,
        ]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcOneAndOdcMoreQualified'
        conditions={[
          `/ctcQualified`,
          `/odcQualified`,
          `/exactlyOneCtcEligibleDependent`,
          `/moreThanOneOdcEligibleDependent`,
        ]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcMoreAndOdcOneQualified'
        conditions={[
          `/ctcQualified`,
          `/odcQualified`,
          `/moreThanOneCtcEligibleDependent`,
          `/exactlyOneOdcEligibleDependent`,
        ]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.ctcMoreAndOdcMoreQualified'
        conditions={[
          `/ctcQualified`,
          `/odcQualified`,
          `/moreThanOneCtcEligibleDependent`,
          `/moreThanOneOdcEligibleDependent`,
        ]}
      />
      <Screen route='ctc-qualified' condition='/ctcQualified'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc' />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/ctc-qualified'
          condition='/hasCtcDependentsCollectionCountGreaterThanOne'
          batches={[`information-architecture-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/ctc-qualified-single'
          condition={{ operator: `isFalse`, condition: `/hasCtcDependentsCollectionCountGreaterThanOne` }}
          batches={[`information-architecture-0`]}
        />
        <ConditionalList
          i18nKey='/info/credits-and-deductions/credits/ctc-qualified'
          items={[{ itemKey: `primary`, collection: `/ctcDependentsCollection` }]}
          batches={[`information-architecture-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/amount-at-end'
          batches={[`information-architecture-0`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <SubSubcategory route='odc-section' headingLevel='h2' borderStyle='heavy'>
      <Gate condition='/flowAskAboutOdc'>
        <Screen route='odc-improper-claims'>
          <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/received-improper-claims'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-mfj'
            condition='/isFilingStatusMFJ'
          />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/potential-credits-improperly-claimed' />
          <InfoDisplay i18nKey='/info/credits-and-deductions/credits/improper-claims' />
          <DFModal i18nKey='/info/credits-and-deductions/credits/improper-claims-odc' />
          <Boolean
            path='/receivedImproperClaims'
            i18nKeySuffixContext='self'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Boolean path='/receivedImproperClaims' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/receivedImproperClaims'>
          <Screen route='odc-improper-claims-taken-again-since-disallowal'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-taken-again'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-taken-again-mfj'
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/potential-credits-improperly-claimed' />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/improper-claims-taken-again-since-disallowal' />
            <Boolean
              path='/hasFiledCtcOdcSinceNoticeExpired'
              i18nKeySuffixContext='self'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Boolean
              path='/hasFiledCtcOdcSinceNoticeExpired'
              i18nKeySuffixContext='mfj'
              condition='/isFilingStatusMFJ'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='odc-improper-claims-notice'
            condition={{ operator: `isFalse`, condition: `/hasFiledCtcOdcSinceNoticeExpired` }}
          >
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice-mfj'
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay i18nKey='/info/credits-and-deductions/credits/potential-credits-improperly-claimed' />
            <DFModal i18nKey='/info/credits-and-deductions/credits/improper-claims-notice' />
            <Enum
              path='/receivedImproperClaimsNotice'
              i18nKeySuffixContext='self'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Enum path='/receivedImproperClaimsNotice' i18nKeySuffixContext='mfj' condition='/isFilingStatusMFJ' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='odc-improper-claims-waiting-period' condition='/hadCtcOdcImproperClaimsBan'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/credits-and-deductions/credits/ctc-or-odc' />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice-expired-mfj'
              condition='/isFilingStatusMFJ'
            />
            <Heading
              i18nKey='/heading/credits-and-deductions/credits/received-improper-claims-notice-expired'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Boolean path='/improperClaimsNoticeExpired' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='odc-outcome-no' headingLevel='h3' editable={false}>
      <Gate condition='/flowAskAboutOdc'>
        <Screen route='odc-not-qualified' condition='/odcHasImproperClaimsAndMaybeHasAdditionalCredits'>
          <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
          <Heading i18nKey='/heading/credits-and-deductions/credits/improper-claims-not-qualified' />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/improper-claims-not-qualified-explanation'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/credits/improper-claims-not-qualified-explanation-mfj'
            condition='/isFilingStatusMFJ'
          />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='odc-outcome-yes' headingLevel='h3' editable={false}>
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.odcQualifiedOne'
        conditions={[
          `/odcQualified`,
          { operator: `isFalse`, condition: `/ctcQualified` },
          `/exactlyOneOdcEligibleDependent`,
        ]}
      />
      <Assertion
        type={`success`}
        i18nKey='dataviews./flow/credits-and-deductions/credits.assertions.odcQualifiedMore'
        conditions={[
          `/odcQualified`,
          { operator: `isFalse`, condition: `/ctcQualified` },
          `/moreThanOneOdcEligibleDependent`,
        ]}
      />
      <Screen route='odc-qualified' condition='/odcQualified'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/odc'
          batches={[`information-architecture-0`]}
        />
        {/* cyc qualified and more than one odc */}
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/odc-qualified-also'
          conditions={[`/ctcQualified`, `/moreThanOneOdcEligibleDependent`]}
          batches={[`information-architecture-0`]}
        />
        {/* ctc qualified and one odc */}
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/odc-qualified-also-single'
          conditions={[`/ctcQualified`, { operator: `isFalse`, condition: `/moreThanOneOdcEligibleDependent` }]}
          batches={[`information-architecture-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/odc-qualified'
          conditions={[{ operator: `isFalse`, condition: `/ctcQualified` }, `/moreThanOneOdcEligibleDependent`]}
          batches={[`information-architecture-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/odc-qualified-single'
          conditions={[
            { operator: `isFalse`, condition: `/ctcQualified` },
            { operator: `isFalse`, condition: `/moreThanOneOdcEligibleDependent` },
          ]}
          batches={[`information-architecture-0`]}
        />
        <ConditionalList
          i18nKey='/info/credits-and-deductions/credits/odc-qualified'
          items={[{ itemKey: `primary`, collection: `/odcDependentsCollection` }]}
          batches={[`information-architecture-0`]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/amount-at-end'
          batches={[`information-architecture-0`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
  </Gate>
);
