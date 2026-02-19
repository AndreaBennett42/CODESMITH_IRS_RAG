/* eslint-disable max-len */
import { Assertion, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ContextHeading,
  DatePicker,
  DFModal,
  DFAlert,
  Enum,
  TaxReturnAlert,
  GenericString,
  Heading,
  HelpLink,
  IconDisplay,
  InfoDisplay,
  IntroContent,
  IpPin,
  LimitingString,
  SaveAndOrContinueButton,
  Tin,
  KnockoutButton,
  MefAlert,
  DFAccordion,
  MultiEnum,
  SetFactAction,
} from '../../ContentDeclarations.js';

export const SpouseSubcategory = (
  <Subcategory
    route='spouse'
    completeIf='/spouseSectionCompleted'
    collectionContext='/secondaryFiler'
    dataItems={[
      {
        itemKey: `maritalStatus`,
      },
      {
        itemKey: `spouse`,
        conditions: [`/isMarried`],
      },
    ]}
  >
    <Assertion
      type='info'
      i18nKey='dataviews./flow/you-and-your-family/spouse.spouseMfjAssertion'
      conditions={[
        `/maritalStatusAllowsFilingMarried`,
        `/wantsJointReturn`,
        { operator: `isIncomplete`, condition: `/filingStatus` },
      ]}
      editRoute='/flow/you-and-your-family/spouse/filing-status-choice-a'
    />
    <Assertion
      type='info'
      i18nKey='dataviews./flow/you-and-your-family/spouse.spouseMfjAssertion'
      conditions={[`/isFilingStatusMFJ`, { operator: `isComplete`, condition: `/filingStatus` }]}
      editRoute='/flow/you-and-your-family/filing-status/filing-status-override'
    />
    <Assertion
      type='info'
      i18nKey='dataviews./flow/you-and-your-family/spouse.spouseMfsAssertion'
      conditions={[`/isMarried`, `/wantsSeparateReturn`, { operator: `isIncomplete`, condition: `/filingStatus` }]}
      editRoute='/flow/you-and-your-family/spouse/filing-status-choice-a'
    />
    <Assertion
      type='info'
      i18nKey='dataviews./flow/you-and-your-family/spouse.spouseMfsAssertion'
      conditions={[`/isFilingStatusMFS`, { operator: `isComplete`, condition: `/filingStatus` }]}
      editRoute='/flow/you-and-your-family/filing-status/filing-status-override'
    />
    <Assertion
      type='info'
      i18nKey='dataviews./flow/you-and-your-family/spouse.waitAssertion'
      conditions={[
        `/wantsHeadOfHousehold`,
        { operator: `isIncomplete`, condition: `/filingStatus` },
        `/eligibleForMFJ`,
      ]}
      editRoute='/flow/you-and-your-family/spouse/filing-status-choice-a'
    />
    <Assertion
      type='info'
      i18nKey='dataviews./flow/you-and-your-family/spouse.hohAssertion'
      conditions={[`/isMarried`, `/isFilingStatusHOH`, { operator: `isComplete`, condition: `/filingStatus` }]}
      editRoute='/flow/you-and-your-family/filing-status/filing-status-override'
    />
    <Screen route='spouse-intro'>
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/you-and-your-family/spouse' />
      <Heading i18nKey='/heading/you-and-your-family/spouse/intro' />
      <IntroContent i18nKey='/info/you-and-your-family/spouse/intro' />
      <DFModal i18nKey='/info/you-and-your-family/about-you/spouse/intro-difference-marital-and-filing' />
      <SaveAndOrContinueButton />
    </Screen>
    <SubSubcategory route='your-marital-status'>
      <Screen route='spouse-marital-status'>
        <DFAlert
          i18nKey='/info/you-and-your-family/spouse/may-change-filing-status-warning'
          headingLevel='h2'
          type='warning'
          condition={{ operator: `isComplete`, condition: `/filingStatus` }}
        />
        <Heading i18nKey='/heading/you-and-your-family/spouse/filing-status' />
        <InfoDisplay i18nKey='/info/you-and-your-family/spouse/marital-status' />
        <DFModal i18nKey='/info/you-and-your-family/spouse/considered-married' />
        <Enum path='/maritalStatus' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='registered-domestic-partner' condition='/checkForRegisteredDomesticPartner'>
        <Heading i18nKey='/heading/you-and-your-family/spouse/registered-domestic-partner' />
        <InfoDisplay i18nKey='/info/you-and-your-family/spouse/registered-domestic-partner' />
        <DFModal i18nKey='/info/you-and-your-family/spouse/what-are-community-property-laws' />
        <Boolean path='/inRegisteredDomesticPartnership' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen
        route='registered-domestic-partner-ko'
        condition='/flowKnockoutHasApplicableRegisteredDomesticPartnership'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/forms-missing/generic-doesnt-have-tax-forms' />
        <InfoDisplay i18nKey='/info/knockout/forms-missing/registered-domestic-partner-ko' />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>

    <SubSubcategory route='your-marital-status'>
      <Gate condition='/isWidowed'>
        <Screen route='spouse-year-of-death'>
          <DFAlert
            i18nKey='/info/you-and-your-family/spouse/may-change-filing-status-warning'
            headingLevel='h2'
            type='warning'
            condition={{ operator: `isComplete`, condition: `/filingStatus` }}
          />
          <Heading i18nKey='/heading/you-and-your-family/spouse/year-of-spouse-death' />
          <InfoDisplay i18nKey='/info/you-and-your-family/spouse/year-of-spouse-death' />
          <Enum path='/yearOfSpouseDeath' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/widowedCouldQualifyForQSS'>
          <Screen route='spouse-widowed-entitled-mfj'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/filed-jointly-year-died' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/filed-jointly-year-died' />
            <DFModal i18nKey='/info/you-and-your-family/spouse/how-to-know-eligible-file-jointly' />
            <Boolean path='/canFileJointlyYearOfSpouseDeath' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </Gate>
    </SubSubcategory>
    <Gate condition='/maritalStatusAllowsFilingMarried'>
      <SubSubcategory route='lived-together-or-apart'>
        <Screen route='spouse-live-together'>
          <Heading i18nKey='/heading/you-and-your-family/spouse/live-together' batches={[`cdcc-0`]} />
          <InfoDisplay i18nKey='/info/you-and-your-family/spouse/live-together' />
          <Boolean path='/livedTogetherAllYearWithSpouse' batches={[`cdcc-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition={{ operator: `isFalse`, condition: `/livedTogetherAllYearWithSpouse` }}>
          <Screen route='spouse-live-together-duration'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/spouse-live-together-months' batches={[`cdcc-0`]} />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/live-together' />
            <Enum path='/spouseLivedTogetherMonths' batches={[`cdcc-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='spouse-apart-last-6-mo' condition='/livedTogetherSixMonthsOrLess'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/live-apart-six-months' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/live-apart' />
            <Boolean path='/writableLivedApartLastSixMonths' batches={[`cdcc-2`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='spouse-living-apart-sep-agreement-last-day'
            condition='/showSpouseLivingApartLastDaySepAgreement'
          >
            <Heading i18nKey='/heading/you-and-your-family/spouse/living-apart-agreement-last-day' />
            <Boolean path='/writableSeparationAgreement' batches={[`cdcc-2`]} />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </SubSubcategory>
      <SubSubcategory route='spouse-citizenship-or-residency'>
        <Assertion
          type='info'
          i18nKey='dataviews./flow/you-and-your-family/spouse.spouseNraMfsAssertion'
          conditions={[
            `/isMarried`,
            { operator: `isFalse`, condition: `/eligibleForMFJ` },
            { operator: `isIncomplete`, condition: `/filingStatus` },
          ]}
        />

        <Screen route='spouse-citizenship'>
          <Heading i18nKey='/heading/you-and-your-family/spouse/citizenship' />
          <DFModal i18nKey='/info/you-and-your-family/spouse/why-citizenship' />
          <Boolean path='/filers/*/isUsCitizenFullYear' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition={{ operator: `isFalse`, condition: `/filers/*/isUsCitizenFullYear` }}>
          <Screen route='spouse-citizen-by-end-ty'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/citizen-end-ty' />
            <DFModal i18nKey='/info/you-and-your-family/what-is-us-resident' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/citizen-end-ty' />
            <Boolean path='/filers/*/writableCitizenAtEndOfTaxYear' />
            <SaveAndOrContinueButton />
          </Screen>
          <Gate condition={{ operator: `isFalse`, condition: `/filers/*/citizenAtEndOfTaxYear` }}>
            <Screen route='spouse-residency'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/resident' />
              <DFModal i18nKey='/info/you-and-your-family/what-is-us-resident' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/an-election' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/resident' />
              <Boolean path='/filers/*/writableIsNoncitizenResidentFullYear' />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition='/filers/*/isNoncitizenResidentFullYear'>
              <Screen route='spouse-national'>
                <Heading i18nKey='/heading/you-and-your-family/spouse/national' />
                <DFModal i18nKey='/info/you-and-your-family/what-is-a-national' />
                <Boolean path='/filers/*/writableIsNational' />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
          </Gate>
        </Gate>
      </SubSubcategory>
      <Gate condition='/readyToEnterSpouseInformation'>
        <SubSubcategory route='filing-status' hidden={true}>
          <Screen route='filing-status-choice-a' condition='/flowShowFilingStatusChoiceA'>
            <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
            <Heading i18nKey='/heading/you-and-your-family/spouse/filing-status-choice-a' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/filing-status-choice-a' />
            <DFModal i18nKey='/info/you-and-your-family/spouse/learn-about-filing-status-i-may-qualify-for' />
            <DFModal i18nKey='/info/you-and-your-family/spouse/what-to-consider-when-mfj-or-mfs' />
            <Enum path='/filingStatusChoice' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='filing-status-choice-b' condition='/flowShowFilingStatusChoiceB'>
            <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
            <Heading i18nKey='/heading/you-and-your-family/spouse/filing-status-choice-b' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/filing-status-choice-b' />
            <DFAlert i18nKey='/info/you-and-your-family/spouse/mfj-more-advantageous' headingLevel='h2' type='info'>
              <DFModal i18nKey='/info/you-and-your-family/spouse/mfj-more-advantageous-modal' />
            </DFAlert>
            <Enum path='/filingStatusChoice' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <Screen
          route='community-property-ko-go-back'
          condition='/knockoutSubjectToCommunityPropertyLaws'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/generic-doesnt-have-tax-forms' />
          <InfoDisplay i18nKey='/info/knockout/forms-missing/community-property-ko' />
          <DFAlert i18nKey='/info/knockout/community-property-ko-go-back' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>

        {/* Married filing jointly section */}

        <Gate condition='/wantsJointReturn'>
          <Screen route='add-spouse-a'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/you-and-your-family/spouse/add-spouse' />
            <Heading i18nKey='/heading/you-and-your-family/spouse/add-spouse-a' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/add-spouse-a' />
            <SaveAndOrContinueButton i18nKey='button.add-spouse' />
          </Screen>
          <SubSubcategory route='spouse-state-residency-and-income'>
            <Screen route='spouse-state-residency-scope'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/residency-scope' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/spouse-state-residency-scope' />
              <Enum path='/spouseLivesInTPState' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='spouse-state-income-form'
              condition={{ operator: `isFalse`, condition: `/spouseResidenceAndIncomeStateOutOfScope` }}
            >
              <Heading i18nKey='/heading/you-and-your-family/spouse/supported-income' batches={[`schedule-b-0`]} />
              <DFModal i18nKey='/info/you-and-your-family/spouse/supported-income' batches={[`schedule-b-0`]} />
              <Enum path='/spouseW2And1099IntInScopedState' batches={[`schedule-b-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='spouse-pfd-income'
              condition={{ operator: `isFalse`, condition: `/spouseLivesOrHasW2InAnotherState` }}
            >
              <Heading
                i18nKey='/heading/you-and-your-family/spouse/state-income-1099-misc-pfd'
                batches={[`alaska-permanent-fund-0`]}
              />
              <InfoDisplay
                i18nKey='/info/you-and-your-family/spouse/state-income-1099-misc-pfd'
                batches={[`alaska-permanent-fund-0`]}
              />
              <Boolean path='/spouseReceivedAlaskaPfd' batches={[`alaska-permanent-fund-0`]} />
              <SaveAndOrContinueButton />
            </Screen>

            <Screen route='spouse-state-income-scope-ko' condition='/flowKnockoutUnsupportedState' isKnockout={true}>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading
                i18nKey='/heading/knockout/forms-missing/supported-state'
                condition={{ operator: `isFalseOrIncomplete`, condition: `/isMarried` }}
              />
              <Heading i18nKey='/heading/knockout/forms-missing/supported-spouse-state' condition='/isMarried' />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
          </SubSubcategory>
          <Screen route='spouse-you-state-scope-confirm'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/confirm' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='spouse-mfj-intro'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/mfj-data-intro' />
            <InfoDisplay i18nKey='/info/you-and-you-family/spouse/mfj-data-intro' />
            <SaveAndOrContinueButton />
          </Screen>
          <SubSubcategory route='spouse-basic-info'>
            <Screen route='spouse-mfj-basic-info'>
              <MefAlert
                type='warning'
                mefErrorCode='F1040-526-03'
                i18nKey='date-of-birth'
                factPaths={[`/filers/*/dateOfBirth`]}
              />
              <MefAlert
                type='warning'
                mefErrorCode='R0000-503-02'
                i18nKey='full-name'
                factPaths={[`/filers/*/fullName`]}
                internalLink='/flow/you-and-your-family/spouse/spouse-mfj-tax-id'
              />
              <Heading i18nKey='/heading/you-and-your-family/spouse/basic-info' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/fill-in-their-name' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/what-if-legal-name-changed' />
              <LimitingString path='/filers/*/firstName' displayOnlyOn='edit' />
              <LimitingString path='/filers/*/writableMiddleInitial' displayOnlyOn='edit' required={false} />
              <LimitingString path='/filers/*/lastName' displayOnlyOn='edit' />
              <Enum path='/filers/*/writableSuffix' displayOnlyOn='edit' required={false} skipBlank={true} />
              <LimitingString path='/filers/*/fullName' displayOnlyOn='data-view' />
              <DatePicker path='/filers/*/dateOfBirth' />
              <DatePicker path='/secondaryFilerDateOfDeath' condition='/isWidowedInTaxYear' lockYearTo='/taxYear' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/why-ask-spouses-age' condition='/isWidowedInTaxYear' />
              <GenericString path='/filers/*/occupation' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/what-occupation' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='spouse-age-ko' condition='/secondaryFiler/youngerThan16' isKnockout={true}>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/forms-missing/filer-age-under-16' />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
          </SubSubcategory>
          <SubSubcategory route='spouse-tax-id'>
            <Screen route='spouse-mfj-tax-id'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/tax-id' />
              <TaxReturnAlert
                type='error'
                i18nKey='/info/non-unique-tin'
                conditions={[
                  `/wantsJointReturn`,
                  { operator: `isComplete`, condition: `/secondaryFiler/tin` },
                  { operator: `isFalse`, condition: `/filers/*/isTinUnique` },
                ]}
              />
              <MefAlert
                type='warning'
                mefErrorCode='R0000-503-02'
                i18nKey='ssn-or-itin'
                internalLink='/flow/you-and-your-family/spouse/spouse-mfj-basic-info'
              />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/tax-id' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/dont-know-or-have-ssn-itin' />
              <Tin path='/secondaryFiler/tin' isSensitive={true} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='spouse-mfj-ssn-valid-for-work' condition='/secondaryFiler/needsSSNWorkStatus'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/ssn-valid-for-work' />
              <DFModal i18nKey='/info/you-and-your-family/ssn-valid-for-work/where' />
              <DFModal i18nKey='/info/you-and-your-family/ssn-valid-for-work/needs-to-be-updated-spouse' />
              <Enum path='/secondaryFilerSsnEmploymentValidity' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <SubSubcategory route='spouse-ssn-federal-benefits'>
            <Screen route='spouse-mfj-ssn-federal-benefits' condition='/secondaryFiler/ssnNotValidForEmployment'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/ssn-federal-benefits' />
              <Boolean path='/secondaryFiler/writableHasSSNOnlyForBenefits' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <SubSubcategory route='spouse-ip-pin'>
            <Screen route='spouse-mfj-ip-pin-choice'>
              <MefAlert
                i18nKey='spouse-ip-pin-choice'
                type='error'
                mefErrorCode='IND-183-01'
                condition={{ operator: `isFalse`, condition: `/filers/*/hasIpPin` }}
              />
              <MefAlert
                i18nKey='spouse-ip-pin-choice'
                type='error'
                mefErrorCode='IND-182-01'
                condition={{ operator: `isFalse`, condition: `/filers/*/hasIpPin` }}
              />
              <Heading i18nKey='/heading/you-and-your-family/spouse/ip-pin-choice' />
              <DFModal i18nKey='/info/ip-pin-choice/what' />
              <InfoDisplay i18nKey='/info/ip-pin-choice' />
              <Boolean path='/filers/*/hasIpPin' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <Gate condition='/filers/*/hasIpPin'>
            <SubSubcategory route='spouse-ip-pin'>
              <Screen route='spouse-mfj-ip-pin-ready'>
                <TaxReturnAlert
                  i18nKey={`/info/you-and-your-family/spouse/ip-pin-not-ready`}
                  conditions={[
                    `/filers/*/hasIpPin`,
                    { operator: `isFalse`, condition: `/filers/*/flowSpouseIpPinReady` },
                  ]}
                  type='warning'
                  headingLevel='h2'
                >
                  <DFAccordion
                    i18nKey='/info/you-and-your-family/spouse/ip-pin-not-ready-explainer'
                    internalLink='/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-ready'
                    headingLevel='h3'
                  />
                </TaxReturnAlert>
                <MefAlert
                  i18nKey='spouse-ip-pin-ready'
                  type='error'
                  mefErrorCode='IND-183-01'
                  conditions={[
                    `/filers/*/hasIpPin`,
                    { operator: `isFalseOrIncomplete`, condition: `/filers/*/flowSpouseIpPinReady` },
                  ]}
                />
                <MefAlert
                  i18nKey='spouse-ip-pin-ready'
                  type='error'
                  mefErrorCode='IND-182-01'
                  conditions={[
                    `/filers/*/hasIpPin`,
                    { operator: `isFalseOrIncomplete`, condition: `/filers/*/flowSpouseIpPinReady` },
                  ]}
                />
                <Heading i18nKey='/heading/you-and-your-family/spouse/ip-pin-ready' />
                <HelpLink i18nKey='/info/ip-pin-ready' />
                <Boolean path='/filers/*/flowSpouseIpPinReady' />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen
                route='spouse-mfj-ip-pin-not-ready'
                condition={{ operator: `isFalse`, condition: `/filers/*/flowSpouseIpPinReady` }}
              >
                <Heading i18nKey='/heading/you-and-your-family/spouse/ip-pin-not-ready' />
                <DFAlert
                  i18nKey='/info/you-and-your-family/spouse/ip-pin-not-ready-assertion'
                  headingLevel='h3'
                  type='warning'
                  internalLink='/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-ready'
                />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
            <SubSubcategory route='spouse-ip-pin'>
              <Screen route='spouse-mfj-ip-pin-input' condition='/filers/*/flowSpouseIpPinReady'>
                <MefAlert
                  i18nKey='spouse-ip-pin-input'
                  type='warning'
                  mefErrorCode='IND-183-01'
                  internalLink='spouse-mfj-ip-pin-ready'
                  conditions={[`/filers/*/hasIpPin`, `/filers/*/flowSpouseIpPinReady`]}
                />
                <MefAlert
                  i18nKey='spouse-ip-pin-input'
                  type='warning'
                  mefErrorCode='IND-182-01'
                  internalLink='spouse-mfj-ip-pin-ready'
                />
                <Heading i18nKey='/heading/you-and-your-family/spouse/ip-pin-input' />
                <IpPin path='/filers/*/identityPin' isSensitive={true} />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
          </Gate>
          <SubSubcategory route='spouse-life-circumstances'>
            <Screen route='spouse-mfj-blind'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/blind' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/blind' condition='/isWidowedInTaxYear' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/blind/defined' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/why-ask-blind' />
              <Boolean path='/filers/*/isBlind' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='spouse-mfj-self-care'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/self-care' batches={[`cdcc-0`]} />
              <DFModal i18nKey='/info/you-and-your-family/spouse/self-care' />
              <Boolean path='/filers/*/isDisabled' batches={[`cdcc-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='spouse-mfj-ft-student'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/isStudent' batches={[`cdcc-0`]} />
              <DFModal
                i18nKey='/info/you-and-your-family/dependents/add-person-student-full-time'
                batches={[`cdcc-0`]}
              />
              <Boolean path='/filers/*/isStudent' batches={[`cdcc-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <SubSubcategory route='taxpayer-dependency-status'>
            <Screen route='spouse-mfj-could-be-claimed'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/can-be-claimed' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/how-do-i-know-spouse-dependent' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/can-be-claimed' />
              <Boolean path='/filers/*/canBeClaimed' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <Gate condition='/flowMFJSpouseFilingRequirementSubsection'>
            <SubSubcategory route='taxpayer-dependency-status'>
              <Screen route='spouse-mfj-spouse-filing-requirement'>
                <Heading i18nKey='/heading/you-and-your-family/spouse/required-to-file' />
                <HelpLink i18nKey='/info/you-and-your-family/spouse/how-do-we-know-if-required-to-file-tax-return' />
                <Boolean path='/MFJRequiredToFile' />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
            <Gate condition='/MFJNotRequiredToFileAndEitherFilerCanBeClaimed'>
              <SubSubcategory route='taxpayer-dependency-status'>
                <Screen route='spouse-mfj-dep-tp-intro'>
                  <Heading i18nKey='/heading/you-and-your-family/spouse/spouse-mfj-dep-tp-intro' batches={[`ptc-1`]} />
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/spouse/spouse-mfj-dep-tp-intro-option-a'
                    batches={[`ptc-1`]}
                  />
                  <DFModal
                    i18nKey='/info/you-and-your-family/spouse/spouse-mfj-dep-tp-intro-what-if-enrolled-MHP'
                    batches={[`ptc-3`]}
                  />
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/spouse/spouse-mfj-dep-tp-intro-option-b'
                    batches={[`ptc-1`]}
                  />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen route='spouse-mfj-refund-only'>
                  <TaxReturnAlert
                    i18nKey='/info/you-and-your-family/spouse/can-not-file-as-dependent-taxpayers'
                    headingLevel='h2'
                    type='error'
                    internalLink='/data-view/flow/you-and-your-family/spouse/MFJDependentsFilingForCredits'
                    condition='/mfjdSwitchNotDependentTP'
                  />
                  <Heading i18nKey='/heading/you-and-your-family/spouse/spouse-mfj-refund-only' batches={[`ptc-1`]} />
                  <InfoDisplay i18nKey='/info/you-and-your-family/spouse/spouse-mfj-refund-only' batches={[`ptc-1`]} />
                  <DFModal
                    i18nKey='/info/you-and-your-family/spouse/impact-claimed-as-dependents'
                    batches={[`ptc-3`]}
                  />
                  <Boolean path='/MFJDependentsFilingForCredits' batches={[`ptc-1`]} />
                  <SaveAndOrContinueButton />
                </Screen>

                <Gate condition={{ operator: `isFalse`, condition: `/MFJDependentsFilingForCredits` }}>
                  <Screen route='spouse-mfj-will-be-claimed' condition='/secondaryFiler/canBeClaimed'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/will-be-claimed' batches={[`updates-0`]} />
                    <InfoDisplay i18nKey='/info/you-and-your-family/spouse/will-be-claimed' batches={[`updates-0`]} />
                    <Boolean path='/filers/*/willBeClaimed' batches={[`updates-0`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-breather'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-breather' batches={[`ptc-1`]} />
                    <DFModal i18nKey='/info/you-and-your-family/spouse/what-is-a-dependent' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-marketplace-plan'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-marketplace-plan' batches={[`ptc-1`]} />
                    <InfoDisplay i18nKey='/info/you-and-your-family/spouse/mfjd-marketplace-plan' batches={[`ptc-1`]} />
                    <DFModal
                      i18nKey='/info/you-and-your-family/spouse/what-is-qual-marketplace-plan'
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
                    <Boolean path='/MFJDepedentsEnrolledMarketplacePlan' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-aptc-paid' condition='/MFJDepedentsEnrolledMarketplacePlan'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-aptc-paid' batches={[`ptc-1`]} />
                    <DFModal i18nKey='/info/you-and-your-family/spouse/ptc-advanced-payments' batches={[`ptc-1`]} />
                    <Boolean path='/advancedPTCPaymentsMade' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-ok-1' condition='/MFJDNotEnrolledMarketplacePlanAndNoAdvancedPTCPaymentsMade'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-ok' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen
                    route='mfjd-enrollees'
                    condition='/refundOnlyAndMFJDEnrolledMarketplacePlanAndAdvancedPTCPaymentsMade'
                  >
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-enrollees' batches={[`ptc-1`]} />
                    <DFModal i18nKey='/info/you-and-your-family/spouse/enrollees-aptc' batches={[`ptc-1`]} />
                    <MultiEnum path='/mfjdEnrollees' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen
                    route='mfjd-other-tax-family'
                    condition='/refundOnlyMFJDEnrollMarketPlanAPTCPaymentsMadeEnrolleeIsOther'
                  >
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-other-tax-family' batches={[`ptc-1`]} />
                    <DFModal i18nKey='/info/credits-and-deductions/credits/what-is-tax-family' batches={[`ptc-1`]} />
                    <InfoDisplay
                      i18nKey='/info/credits-and-deductions/credits/other-tax-family-guidance'
                      batches={[`ptc-1`]}
                    />
                    <DFModal
                      i18nKey='/info/credits-and-deductions/credits/what-if-enrolled-more-than-one'
                      batches={[`ptc-1`]}
                    />
                    <Boolean path='/mfjdOtherTaxFamily' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-switch-not-dependent-tp' condition='/mfjdSwitchNotDependentTP'>
                    <Heading
                      i18nKey='/heading/you-and-your-family/spouse/mfjd-switch-not-dependent-tp'
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-switch-not-dependent-tp/not-dependent-self-enrollee'
                      conditions={[
                        { operator: `isTrue`, condition: `/primaryFilerHasMarketplacePlanAndIsNotClaimedAsDependent` },
                        {
                          operator: `isFalse`,
                          condition: `/secondaryFilerHasMarketplacePlanAndIsNotClaimedAsDependent`,
                        },
                      ]}
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-switch-not-dependent-tp/not-dependent-spouse-enrollee'
                      conditions={[
                        { operator: `isFalse`, condition: `/primaryFilerHasMarketplacePlanAndIsNotClaimedAsDependent` },
                        {
                          operator: `isTrue`,
                          condition: `/secondaryFilerHasMarketplacePlanAndIsNotClaimedAsDependent`,
                        },
                      ]}
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-switch-not-dependent-tp/not-dependent-self-and-spouse-enrollee'
                      conditions={[
                        { operator: `isTrue`, condition: `/primaryFilerHasMarketplacePlanAndIsNotClaimedAsDependent` },
                        {
                          operator: `isTrue`,
                          condition: `/secondaryFilerHasMarketplacePlanAndIsNotClaimedAsDependent`,
                        },
                      ]}
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-switch-not-dependent-tp/not-other-tax-family'
                      condition='/notDependentNotOtherTaxFamily'
                      batches={[`ptc-1`]}
                    />
                    <SetFactAction path='/showMFJDChoiceBDynamic' source='/flowFalse' />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-ok-2' condition='/mfjdOK'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/mfjd-ok' batches={[`ptc-1`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='mfjd-ok-form-not-required' condition='/mfjdOkFormNotRequired'>
                    <Heading
                      i18nKey='/heading/you-and-your-family/spouse/mfjd-ok-form-not-required'
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-ok-form-not-required/primaryOnly'
                      condition='/dependentClaimedAndEnrolleeIsSelf'
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-ok-form-not-required/secondaryOnly'
                      condition='/dependentClaimedAndEnrolleeIsSpouse'
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-ok-form-not-required/primaryAndSecondary'
                      condition='/dependentClaimedAndEnrolleeIsBoth'
                      batches={[`ptc-1`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/mfjd-ok-form-not-required/claimer-needs-1095A'
                      batches={[`ptc-1`]}
                    />
                    <SaveAndOrContinueButton />
                  </Screen>
                </Gate>
                <Screen route='mfj-dependent-choice-a' condition='/mfjDependentChoiceA'>
                  <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
                  <Heading i18nKey='/heading/you-and-your-family/spouse/mfj-dependent-choice-a' batches={[`ptc-1`]} />
                  <InfoDisplay i18nKey='/info/you-and-your-family/spouse/mfj-dependent-choice-a' batches={[`ptc-3`]} />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen
                  route='mfj-dependent-choice-b'
                  condition={{ operator: `isFalse`, condition: `/MFJClaimingRefundOnly` }}
                >
                  <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' batches={[`ptc-1`]} />
                  <Heading i18nKey='/heading/you-and-your-family/spouse/mfj-dependent-choice-b' batches={[`ptc-1`]} />
                  <InfoDisplay i18nKey='/info/you-and-your-family/spouse/mfj-dependent-choice-b' batches={[`ptc-3`]} />
                  <InfoDisplay
                    i18nKey='/info/you-and-your-family/spouse/mfj-dependent-choice-b-dynamic'
                    condition='/showMFJDChoiceBDynamic'
                    batches={[`ptc-1`]}
                  />
                  <DFAlert
                    i18nKey='/info/will-be-claimed-cant-be-claimed-contradiction'
                    headingLevel='h2'
                    type='error'
                    conditions={[
                      `/primaryFiler/willBeClaimed`,
                      `/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
                    ]}
                    batches={[`updates-0`]}
                  />
                  <SaveAndOrContinueButton />
                </Screen>
              </SubSubcategory>
            </Gate>
          </Gate>
        </Gate>

        {/* Married filing separately section */}

        <Gate condition='/wantsSeparateReturn'>
          <Screen route='add-spouse-b'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/you-and-your-family/spouse/add-spouse' />
            <Heading i18nKey='/heading/you-and-your-family/spouse/add-spouse-b' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/add-spouse-b' />
            <SaveAndOrContinueButton i18nKey='button.add-spouse' />
          </Screen>
          <Screen route='mfs-spouse-data-intro'>
            <Heading i18nKey='/heading/you-and-your-family/spouse/mfs-data-intro' />
            <InfoDisplay i18nKey='/info/you-and-your-family/spouse/mfs-data-intro' />
            <SaveAndOrContinueButton />
          </Screen>
          <SubSubcategory route='spouse-separate-return'>
            <Screen
              route='mfs-spouse-filing-a-return-living'
              condition={{ operator: `isFalse`, condition: `/isWidowed` }}
            >
              <Heading i18nKey='/heading/you-and-your-family/spouse/filing-a-return-living' />
              <Boolean path='/MFSLivingSpouseFilingReturn' displayOnlyOn='edit' />
              <Boolean path='/MFSLivingSpouseFilingReturn' displayOnlyOn='data-view' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='mfs-spouse-filing-a-return-widowed' condition='/isWidowed'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/filing-a-return-widowed' />
              <Boolean path='/MFSDeceasedSpouseFilingReturn' displayOnlyOn='edit' />
              <Boolean path='/MFSDeceasedSpouseFilingReturn' displayOnlyOn='data-view' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='mfs-spouse-itemizing' condition='/MFSSpouseFilingReturnDerived'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/itemizing' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/itemizing' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/itemizing-difference' />
              <Boolean path='/spouseItemizes' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='mfs-spouse-itemizing-ko' condition='/spouseItemizesKnockout' isKnockout={true}>
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/forms-missing/generic-doesnt-have-tax-forms' />
              <InfoDisplay i18nKey='/info/knockout/spouse-itemizes' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/itemizing-difference' />
              <DFAlert i18nKey='/info/knockout/spouse-itemizes' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
          </SubSubcategory>
          <SubSubcategory route='spouse-basic-info'>
            <Screen route='mfs-spouse-name'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/name' />
              <LimitingString path='/filers/*/firstName' displayOnlyOn='edit' />
              <LimitingString path='/filers/*/writableMiddleInitial' displayOnlyOn='edit' required={false} />
              <LimitingString path='/filers/*/lastName' displayOnlyOn='edit' />
              <LimitingString path='/filers/*/fullName' displayOnlyOn='data-view' />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>
          <SubSubcategory route='spouse-tax-id'>
            <Screen
              route='mfs-spouse-nr-tin'
              condition={{ operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/isUsCitizenFullYear` }}
            >
              <Heading i18nKey='/heading/you-and-your-family/spouse/nrtin' />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/nrtin' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/who-needs-to-file-tax-return' />
              <Boolean path='/MFSSpouseHasNRTIN' />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/secondaryFiler/isUsCitizenFullYear` }}>
              <Screen route='mfs-breather-spouse-nr-tin' condition='/MFSLivesInMdNcSpouseHasNoTin'>
                <Heading i18nKey='/heading/you-and-your-family/spouse/nrtin-breather' />
                <InfoDisplay i18nKey='/info/you-and-your-family/spouse/nrtin-breather' />
                <DFModal
                  i18nKey='/info/you-and-your-family/spouse/nrtin-breather-how-to-file-paper-md'
                  condition={{ operator: `isTrue`, condition: `/livedInMd` }}
                />
                <DFModal
                  i18nKey='/info/you-and-your-family/spouse/nrtin-breather-how-to-file-paper-nc'
                  condition={{ operator: `isTrue`, condition: `/livedInNc` }}
                />
                <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file-help' headingLevel='h2' type='info' />
                <SaveAndOrContinueButton i18nKey='button.continueWoSpouseTin' />
              </Screen>
            </Gate>

            <Screen route='mfs-spouse-tax-id' condition='/MFSNeedSpouseTaxId'>
              <Heading i18nKey='/heading/you-and-your-family/spouse/tax-id' />
              <TaxReturnAlert
                type='error'
                i18nKey='/info/non-unique-tin'
                conditions={[
                  `/wantsSeparateReturn`,
                  { operator: `isComplete`, condition: `/secondaryFiler/tin` },
                  { operator: `isFalse`, condition: `/filers/*/isTinUnique` },
                ]}
              />
              <InfoDisplay i18nKey='/info/you-and-your-family/spouse/tax-id' />
              <DFModal i18nKey='/info/you-and-your-family/spouse/dont-know-or-have-ssn-itin' />
              <Tin path='/secondaryFiler/tin' isSensitive={true} />
              <SaveAndOrContinueButton />
            </Screen>
          </SubSubcategory>

          {/* If we learn that the MFS spouse is ineligible for providing additional standard deduction benefits, we
          stop asking questions about them and never get their age/vision situation  */}
          <Gate condition={{ operator: `isFalse`, condition: `/MFSSpouseFilingReturnDerived` }}>
            <SubSubcategory route='spouse-taxpayer-dependency-status'>
              <Screen route='mfs-spouse-gross-income'>
                <Heading i18nKey='/heading/you-and-your-family/spouse/gross-income' />
                <InfoDisplay i18nKey='/info/you-and-your-family/spouse/gross-income' />
                <HelpLink i18nKey='/info/you-and-your-family/spouse/learn-more-gross-income' />
                <Boolean path='/MFSSpouseHasGrossIncome' />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
            <Gate condition={{ operator: `isFalse`, condition: `/MFSSpouseHasGrossIncome` }}>
              <SubSubcategory route='spouse-taxpayer-dependency-status'>
                <Screen route='mfs-spouse-could-be-claimed'>
                  <Heading i18nKey='/heading/you-and-your-family/spouse/can-be-claimed' />
                  <DFModal i18nKey='/info/you-and-your-family/spouse/how-do-i-know-spouse-dependent' />
                  <InfoDisplay i18nKey='/info/you-and-your-family/spouse/can-be-claimed' />
                  <Boolean path='/secondaryFiler/canBeClaimed' />
                  <SaveAndOrContinueButton />
                </Screen>
              </SubSubcategory>
              <Gate condition={{ operator: `isFalse`, condition: `/secondaryFiler/canBeClaimed` }}>
                <SubSubcategory route='spouse-age-vision'>
                  <Screen route='mfs-spouse-age'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/age' />
                    <InfoDisplay i18nKey='/info/you-and-your-family/spouse/age' />
                    <InfoDisplay
                      i18nKey='/info/you-and-your-family/spouse/age-at-death'
                      condition='/isWidowedInTaxYear'
                    />
                    <DFModal i18nKey='/info/you-and-your-family/spouse/why-ask-spouses-age' />
                    <Boolean path='/MFSSpouse65OrOlder' />
                    <SaveAndOrContinueButton />
                  </Screen>
                </SubSubcategory>
                <SubSubcategory route='spouse-age-vision'>
                  <Screen route='mfs-spouse-blind'>
                    <Heading i18nKey='/heading/you-and-your-family/spouse/blind' />
                    <InfoDisplay i18nKey='/info/you-and-your-family/spouse/blind' condition='/isWidowedInTaxYear' />
                    <DFModal i18nKey='/info/you-and-your-family/spouse/why-ask-blind' />
                    <DFModal i18nKey='/info/you-and-your-family/spouse/blind/defined' />
                    <Boolean path='/secondaryFiler/isBlind' />
                    <SaveAndOrContinueButton />
                  </Screen>
                </SubSubcategory>
              </Gate>
            </Gate>
          </Gate>
        </Gate>
      </Gate>
    </Gate>
  </Subcategory>
);
