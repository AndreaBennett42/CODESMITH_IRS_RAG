import { Screen, SubSubcategory, Gate, CollectionLoop } from '../../flowDeclarations.js';
import {
  Boolean,
  ConditionalList,
  ContextHeading,
  DFAlert,
  DFModal,
  Heading,
  HelpLink,
  InfoDisplay,
  InternalLink,
  IpPin,
  SaveAndOrContinueButton,
  Tin,
} from '../../ContentDeclarations.js';

export const CdccNonDependentQpTinPinLoop = (
  <Gate condition='/showCdccNondependentQpTinPin'>
    <SubSubcategory route='cdcc-nondependent-qp-tin-pin' headingLevel='h3'>
      <Screen
        route='cdcc-nondependent-qps'
        condition={{ operator: `isFalse`, condition: `/cdccNonDepQpTinsNoneToCollect` }}
      >
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/cdcc-nondep-qps'
          condition={{ operator: `isFalse`, condition: `/hasExactlyOneCdccNonDepQpTinNeeded` }}
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/credits-and-deductions/credits/cdcc-nondep-qps-count-one'
          condition='/hasExactlyOneCdccNonDepQpTinNeeded'
          batches={[`cdcc-0`]}
        />
        <ConditionalList
          i18nKey='/heading/credits-and-deductions/credits/cdcc-nondep-qp-tin-needed'
          batches={[`cdcc-0`]}
          items={[{ itemKey: `primary`, collection: `/cdccNonDependentQualifyingPeopleTinsPinsNeeded` }]}
        />
        <InfoDisplay
          i18nKey='/info/credits-and-deductions/credits/cdcc-nondep-qps-tins-saved'
          condition='/cdccNonDepQpTinsSavedNotZero'
          batches={[`cdcc-0`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </SubSubcategory>
    <CollectionLoop
      collection='/cdccNonDependentQualifyingPeopleAssignedTins'
      loopName='/cdccNonDependentQualifyingPeopleAssignedTins'
      autoIterate={true}
      collectionItemCompletedCondition='/cdccNonDepQpTinsNoneToCollect'
    >
      <SubSubcategory route='cdcc-nondep-qp-info' headingLevel='h3'>
        <Screen route='cdcc-nondep-qp-enter-tin'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/enter-tin' />
          <Tin path='/familyAndHousehold/*/tin' isSensitive={true} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='cdcc-nondep-qp-ip-pin-choice'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-choice' />
          <DFModal batches={[`cdcc-1`]} i18nKey='/info/ip-pin-choice/what' />
          <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/ip-pin-choice' />
          <Boolean path='/familyAndHousehold/*/hasIpPin' batches={[`cdcc-3`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/familyAndHousehold/*/hasIpPin'>
          <Screen route='cdcc-nondep-qp-ip-pin-ready'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/cdcc'
              batches={[`cdcc-0`]}
            />
            <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-ready' />
            <InfoDisplay batches={[`cdcc-1`]} i18nKey='/info/credits-and-deductions/credits/ip-pin-ready' />
            <Boolean path='/familyAndHousehold/*/flowIpPinReady' batches={[`cdcc-3`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='cdcc-nondep-qp-ip-pin-delay' condition='/familyAndHousehold/*/isMissingIpPin'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/cdcc'
              batches={[`cdcc-0`]}
            />
            <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-delay' />
            <HelpLink i18nKey='/info/learn-more-retrieve-ip-pin' />
            <DFAlert i18nKey='/info/ip-pin-not-ready' headingLevel='h3' type='warning' />
            <DFModal i18nKey='/info/file-ip-return-without-ip-pin' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='cdcc-nondep-qp-ip-pin-input' condition='/familyAndHousehold/*/flowIpPinReady'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/cdcc'
              batches={[`cdcc-3`]}
            />
            <Heading batches={[`cdcc-1`]} i18nKey='/heading/credits-and-deductions/credits/ip-pin-input' />
            <IpPin path='/familyAndHousehold/*/identityPin' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
        <Screen route='cdcc-nondep-qp-confirmation'>
          <ContextHeading
            displayOnlyOn='edit'
            i18nKey='/heading/credits-and-deductions/credits/cdcc'
            batches={[`cdcc-0`]}
          />
          <Heading i18nKey='/heading/credits-and-deductions/credits/cdcc-qp-confirmation' batches={[`cdcc-2`]} />
          <InternalLink
            i18nKey='/info/credits-and-deductions/credits/cdcc-nondep-qp-confirmation'
            route='/data-view/flow/you-and-your-family/dependents'
            batches={[`cdcc-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </CollectionLoop>
  </Gate>
);
