/* eslint-disable max-len */
import { Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  DFModal,
  Enum,
  Heading,
  InfoDisplay,
  LimitingString,
  PhoneNumber,
  Pin,
  SaveAndOrContinueButton,
} from '../../ContentDeclarations.js';

export const OtherPreferencesSubcategory = (
  <Subcategory
    route='other-preferences'
    completeIf='/completedOtherPrefsSection'
    dataItems={[
      {
        itemKey: `thirdPartyDesignee`,
        conditions: [`/wantsThirdPartyDesignee`],
      },
      {
        itemKey: `languagePref`,
        conditions: [`/wantsCustomLanguage`],
      },
      {
        itemKey: `presidentialCampaign`,
        conditions: [`/hasPresidentialCampaignAmount`],
      },
    ]}
  >
    <Screen route='other-preferences-intro'>
      {/* TODO: Temporary heading - will be resolved as part of https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/4386 */}
      <Heading i18nKey='/heading/other-preferences' />
      <InfoDisplay
        i18nKey='/info/other-preferences/other-preferences-intro'
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalse`, condition: `/totalTaxIsHigherThanPresidentialCampaignDesignationAmount` },
        ]}
      />
      <InfoDisplay
        i18nKey='/info/other-preferences/other-preferences-intro-with-pres-fund'
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          `/totalTaxIsHigherThanPresidentialCampaignDesignationAmount`,
        ]}
      />
      <InfoDisplay
        i18nKey='/info/other-preferences/other-preferences-intro-mfj'
        conditions={[
          `/isFilingStatusMFJ`,
          { operator: `isFalse`, condition: `/totalTaxIsHigherThanPresidentialCampaignDesignationAmount` },
        ]}
      />
      <InfoDisplay
        i18nKey='/info/other-preferences/other-preferences-intro-mfj-with-pres-fund'
        conditions={[`/isFilingStatusMFJ`, `/totalTaxIsHigherThanPresidentialCampaignDesignationAmount`]}
      />
      <SaveAndOrContinueButton />
    </Screen>
    <SubSubcategory route='create-self-select-pin'>
      <Gate condition='isEssarSigningPath'>
        <Screen route='create-new-self-select-pin'>
          <Heading i18nKey='/heading/other-preferences/create-self-select-pin' />
          <InfoDisplay i18nKey='/info/other-preferences/create-self-select-pin' />
          <Pin path='/selfSelectPin' />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='third-party-designee'>
      <Screen route='third-party-designee-choice'>
        <Heading i18nKey='/heading/other-preferences/other-preferences-third-party-designee-choice' />
        <DFModal i18nKey='/info/other-preferences/other-preferences-third-party-designee-choice' />
        <Boolean path='/wantsThirdPartyDesignee' />
        <SaveAndOrContinueButton />
      </Screen>
      <Gate condition='/wantsThirdPartyDesignee'>
        <Screen route='third-party-designee-input'>
          <Heading i18nKey='/heading/other-preferences/other-preferences-third-party-designee-input' />
          <LimitingString path='/thirdPartyDesigneeFullName' />
          <PhoneNumber path='/thirdPartyDesigneePhone' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='third-party-designee-pin'>
          <Heading i18nKey='/heading/other-preferences/other-preferences-third-party-designee-pin' />
          <InfoDisplay i18nKey='/info/other-preferences/other-preferences-third-party-designee-pin' />
          <Pin path='/thirdPartyDesigneePin' isSensitive={true} />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='irs-communication'>
      <Screen route='comms-different-language-choice'>
        <Heading i18nKey='/heading/other-preferences/other-preferences-comms-language-format-choice' />
        <InfoDisplay
          i18nKey='/info/other-preferences/other-preferences-comms-language-format-choice'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <InfoDisplay
          i18nKey='/info/other-preferences/other-preferences-comms-language-format-choice-mfj'
          condition='/isFilingStatusMFJ'
        />
        <DFModal i18nKey='/info/other-preferences/when-will-irs-start-using-requested-language' />
        <Boolean path='/wantsCustomLanguage' />
        <SaveAndOrContinueButton />
      </Screen>
      <Gate condition='/wantsCustomLanguage'>
        <Screen route='comms-different-language-input'>
          <Heading i18nKey='/heading/other-preferences/other-preferences-comms-different-language-input' />
          <Enum path='/languagePreference' />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
      <Screen route='comms-different-format-choice'>
        <Heading i18nKey='/heading/other-preferences/other-preferences-comms-different-format-choice' />
        <InfoDisplay
          i18nKey='/info/other-preferences/other-preferences-comms-different-format-choice'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <InfoDisplay
          i18nKey='/info/other-preferences/other-preferences-comms-different-format-choice-mfj'
          condition='/isFilingStatusMFJ'
        />
        <Boolean path='/wantsCommsFormat' />
        <SaveAndOrContinueButton />
      </Screen>
      <Gate condition='/wantsCommsFormat'>
        <Screen route='comms-different-format-input'>
          <Heading i18nKey='/heading/other-preferences/other-preferences-comms-different-format-input' />
          <InfoDisplay i18nKey='/info/other-preferences/other-preferences-comms-different-format-input' />
          <Enum path='/commsFormat' />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
    <SubSubcategory route='presidential-fund'>
      <Gate condition='/totalTaxIsHigherThanPresidentialCampaignDesignationAmount'>
        <Screen route='presidential-fund'>
          <Heading
            i18nKey='/heading/other-preferences/other-preferences-presidential-fund-single'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/other-preferences/other-preferences-presidential-fund-married'
            condition='/isFilingStatusMFJ'
          />
          <InfoDisplay i18nKey='/info/other-preferences/other-preferences-presidential-fund' />
          <DFModal i18nKey='/info/other-preferences/other-preferences-presidential-fund/what-is-it' />
          <Enum path='/presidentalCampaignDesignation' />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </SubSubcategory>
  </Subcategory>
);
