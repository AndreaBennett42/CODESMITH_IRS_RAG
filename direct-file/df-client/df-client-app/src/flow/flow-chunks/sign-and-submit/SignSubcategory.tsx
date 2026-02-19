/* eslint-disable max-len */
import { Assertion, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ContextHeading,
  DatePicker,
  DFModal,
  DFAlert,
  Dollar,
  Heading,
  InfoDisplay,
  InternalLink,
  Pin,
  SaveAndOrContinueButton,
  Enum,
  SetFactAction,
  MefAlert,
  DFAccordion,
} from '../../ContentDeclarations.js';

export const SignSubcategory = (
  <Subcategory
    route='sign-and-submit'
    completeIf={`/signSectionComplete`}
    isSignAndSubmit={true}
    displayOnlyIf={[
      { operator: `isFalseOrIncomplete`, condition: `isEssarSigningPath` },
      { operator: `isFalseOrIncomplete`, condition: `/isPaperPathDueToMissingIpPin` },
    ]}
  >
    <Screen route='sign-return-intro'>
      <SetFactAction path='/directFileLanguagePreference' source='df.language' />
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/complete/sign-and-submit/sign-return-intro' />
      <Heading
        i18nKey='/heading/complete/sign-and-submit/intro'
        condition={{ operator: `isFalse`, condition: `/isMFJWithLivingSpouse` }}
      />
      <Heading
        i18nKey='/heading/complete/sign-and-submit/intro-married-filing-jointly'
        condition='/isMFJWithLivingSpouse'
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/intro'
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/intro-married-filing-jointly'
        condition='/isFilingStatusMFJ'
      />
      <DFModal i18nKey='/info/complete/sign-and-submit/where-to-find-agi-or-pin' />
      <SaveAndOrContinueButton />
    </Screen>
    <SubSubcategory route='your-digital-signature'>
      <Assertion type='info' i18nKey='dataviews./flow/complete/sign-and-submit.dateOfBirthAssertion' />
      <Screen
        condition={{ operator: `isFalse`, condition: `/primaryFilerHasIpPin` }}
        route='sign-return-filed-last-year'
      >
        <Heading i18nKey='/heading/complete/sign-and-submit/filed-last-year' />
        <Boolean path='/filedLastYear' />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>

    <Gate condition='/filedLastYear'>
      <SubSubcategory route='your-digital-signature'>
        <Screen route='sign-return-identity'>
          <Heading i18nKey='/heading/complete/sign-and-submit/identity' />
          <MefAlert type='warning' mefErrorCode='IND-031-04' i18nKey='identity' conditions={[`/filedLastYear`]} />
          <InfoDisplay i18nKey='/info/complete/sign-and-submit/matching-warning' />
          <DFModal i18nKey='/info/complete/sign-and-submit/where-to-find-agi-or-pin' />
          <Enum path='/signReturnIdentity' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='sign-paper-file-explain' condition='/cannotFindPinOrAgi'>
          <Heading i18nKey='/heading/complete/sign-and-submit/paper-file-explain' />
          <DFAlert headingLevel='h2' i18nKey='/info/paper-filing' type='warning'>
            <DFAccordion i18nKey='/info/paper-filing-explainer' asExpanded />
          </DFAlert>
          <SetFactAction path='/flowHasSeenPrintAndMail' source='/flowTrue' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition={{ operator: `isFalse`, condition: `/cannotFindPinOrAgi` }}>
          <Screen route='sign-return-enter-self-select-pin' condition='/willEnterLastYearPin'>
            <Heading i18nKey='/heading/complete/sign-and-submit/enter-self-select-pin' />
            <MefAlert
              type='warning'
              mefErrorCode='IND-031-04'
              i18nKey='pin'
              internalLink='/flow/complete/sign-and-submit/sign-return-identity'
              conditions={[{ operator: `isFalse`, condition: `/cannotFindPinOrAgi` }, `/willEnterLastYearPin`]}
            />
            <DFModal i18nKey='/info/complete/sign-and-submit/sign-return-enter-self-select-pin' />
            <Pin path='/selfSelectPinLastYear' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </SubSubcategory>
    </Gate>
    <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/cannotFindPinOrAgi` }}>
      <SubSubcategory route='your-digital-signature'>
        <Screen route='sign-return-agi' condition='/needsToEnterLastYearAgi'>
          <Heading i18nKey='/heading/complete/sign-and-submit/enter-last-year-agi' />
          <MefAlert
            type='warning'
            mefErrorCode='IND-031-04'
            i18nKey='agi'
            internalLink='/flow/complete/sign-and-submit/sign-return-identity'
            conditions={[
              { operator: `isFalseOrIncomplete`, condition: `/cannotFindPinOrAgi` },
              `/needsToEnterLastYearAgi`,
            ]}
          />
          <InfoDisplay
            i18nKey='/info/complete/sign-and-submit/agi-import-explanation'
            condition={{ operator: `isFalse`, condition: `/treatAsMFJ` }}
          />
          <InfoDisplay i18nKey='/info/complete/sign-and-submit/agi-import-explanation-mfj' condition='/treatAsMFJ' />
          <DFModal i18nKey='/info/complete/sign-and-submit/enter-last-year-agi' />
          <Dollar path='/lastYearAgi' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='sign-return-create-new-self-select-pin'>
          <Heading i18nKey='/heading/complete/sign-and-submit/create-new-self-select-pin' />
          <Pin path='/selfSelectPin' />
          <DatePicker
            path='/primaryFiler/dateOfBirth'
            displayOnlyOn='data-view'
            editRoute='/flow/you-and-your-family/about-you/about-you-basic-info'
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </Gate>
    <Gate condition='/isMFJWithLivingSpouse'>
      <Screen route='sign-return-spouse-intro'>
        <Heading i18nKey='/heading/complete/sign-and-submit/sign-return-spouse-intro' />
        <DFModal i18nKey='/info/complete/sign-and-submit/where-to-find-agi-or-pin-they' />
        <InfoDisplay i18nKey='/info/complete/sign-and-submit/sign-return-spouse-intro' />
        <SaveAndOrContinueButton />
      </Screen>
      <SubSubcategory route='spouse-digital-signature'>
        <Screen
          route='sign-return-spouse-filed-last-year'
          condition={{ operator: `isFalse`, condition: `/spouseHasIpPin` }}
        >
          <Heading i18nKey='/heading/complete/sign-and-submit/sign-return-spouse-filed-last-year' />
          <Boolean path='/spouseFiledLastYear' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Gate condition='/spouseFiledLastYear'>
        <SubSubcategory route='spouse-digital-signature'>
          <Assertion
            type='info'
            i18nKey='dataviews./flow/complete/sign-and-submit.dateOfBirthAssertion'
            condition='/isMFJWithLivingSpouse'
          />
          <Screen route='sign-return-spouse-identity'>
            <Heading i18nKey='/heading/complete/sign-and-submit/spouse-identity' />
            <InfoDisplay i18nKey='/info/complete/sign-and-submit/matching-warning-they' />
            <DFModal i18nKey='/info/complete/sign-and-submit/where-to-find-agi-or-pin-they' />
            <MefAlert
              type='warning'
              mefErrorCode='IND-032-04'
              i18nKey='identity'
              conditions={[`/spouseFiledLastYear`]}
            />
            <Enum path='/signReturnIdentitySpouse' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='sign-paper-file-spouse-explain' condition='/spouseCannotFindPinOrAgi'>
            <Heading i18nKey='/heading/complete/sign-and-submit/paper-file-spouse-explain' />
            <DFAlert headingLevel='h2' i18nKey='/info/paper-filing-spouse' type='warning'>
              <DFAccordion i18nKey='/info/paper-filing-spouse-explainer' />
            </DFAlert>
            <SetFactAction path='/flowHasSeenPrintAndMail' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
          <Gate condition={{ operator: `isFalse`, condition: `/spouseCannotFindPinOrAgi` }}>
            <Screen route='sign-return-spouse-enter-self-select-pin' condition='/spouseWillEnterLastYearPin'>
              <Heading i18nKey='/heading/complete/sign-and-submit/sign-return-spouse-enter-self-select-pin' />
              <MefAlert
                type='warning'
                mefErrorCode='IND-032-04'
                i18nKey='pin'
                internalLink='/flow/complete/sign-and-submit/sign-return-spouse-identity'
                conditions={[
                  { operator: `isFalse`, condition: `/spouseCannotFindPinOrAgi` },
                  `/spouseWillEnterLastYearPin`,
                ]}
              />
              <DFModal i18nKey='/info/complete/sign-and-submit/sign-return-enter-self-select-pin-spouse' />
              <Pin path='/spouseSelfSelectPinLastYear' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
        </SubSubcategory>
      </Gate>
      <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/spouseCannotFindPinOrAgi` }}>
        <SubSubcategory route='spouse-digital-signature'>
          <Screen route='sign-return-spouse-agi' condition='/spouseNeedsToEnterLastYearAgi'>
            <Heading i18nKey='/heading/complete/sign-and-submit/sign-return-spouse-agi' />
            <MefAlert
              type='warning'
              mefErrorCode='IND-032-04'
              i18nKey='agi'
              internalLink='/flow/complete/sign-and-submit/sign-return-spouse-identity'
              conditions={[
                { operator: `isFalseOrIncomplete`, condition: `/spouseCannotFindPinOrAgi` },
                `/spouseNeedsToEnterLastYearAgi`,
              ]}
            />
            <InfoDisplay i18nKey='/info/complete/sign-and-submit/agi-import-explanation-mfj' condition='/treatAsMFJ' />
            <DFModal i18nKey='/info/complete/sign-and-submit/sign-return-spouse-agi' />
            <Dollar path='/spouseLastYearAgi' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='sign-return-spouse-create-self-select-pin'>
            <Heading i18nKey='/heading/complete/sign-and-submit/sign-return-spouse-create-new-self-select-pin' />
            <Pin path='/spouseSelfSelectPin' />
            <DatePicker
              path='/secondaryFiler/dateOfBirth'
              displayOnlyOn='data-view'
              editRoute='/flow/you-and-your-family/spouse/spouse-mfj-basic-info'
            />
            <InfoDisplay i18nKey='/info/complete/sign-and-submit/spouse-confirm-identity' />
            <InternalLink
              i18nKey='/info/complete/sign-and-submit/date-of-birth-secondary-filer'
              route='/flow/you-and-your-family/spouse/spouse-mfj-basic-info'
            />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Gate>
    </Gate>
  </Subcategory>
);
