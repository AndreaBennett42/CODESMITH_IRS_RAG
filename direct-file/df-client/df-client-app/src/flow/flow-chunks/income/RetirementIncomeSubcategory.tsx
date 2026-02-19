/* eslint-disable max-len */
import { CollectionLoop, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  CollectionItemManager,
  CollectionItemReference,
  ContextHeading,
  DFAlert,
  DFModal,
  GenericString,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
  LimitingString,
  Dollar,
  Boolean,
  Enum,
  SetFactAction,
  PhoneNumber,
  Address,
  IconDisplay,
  Tin,
  KnockoutButton,
  DatePicker,
  Ein,
  TaxReturnAlert,
} from '../../ContentDeclarations.js';

export const RetirementIncomeSubcategory = (
  <Subcategory
    route='retirement'
    completeIf='/form1099RsIsComplete'
    collectionContext='/form1099Rs'
    displayOnlyIf='/is1099RFeatureFlagEnabled'
    dataItems={[
      {
        itemKey: `retirementTaxable`,
        conditions: [`/has1099R`],
      },
      {
        itemKey: `retirementNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/has1099R` }],
      },
    ]}
  >
    <Screen route='1099-r-loop-intro'>
      <ContextHeading
        displayOnlyOn='edit'
        i18nKey='/heading/income/retirement/retirement-loop'
        batches={[`retirement-1099R-4`]}
        condition={{ operator: `isFalseOrIncomplete`, condition: `/has1099R` }}
      />
      <Heading
        i18nKey='/heading/income/retirement/retirement-loop'
        batches={[`retirement-1099R-4`]}
        condition='/has1099R'
      />

      <Heading
        i18nKey='/heading/income/retirement/retirement-loop-intro'
        batches={[`retirement-1099R-4`]}
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/has1099R` },
        ]}
      />
      <Heading
        i18nKey='/heading/income/retirement/retirement-loop-intro-mfj'
        batches={[`retirement-1099R-4`]}
        conditions={[{ condition: `/isFilingStatusMFJ` }, { operator: `isFalseOrIncomplete`, condition: `/has1099R` }]}
      />

      <DFModal
        i18nKey='/info/income/retirement/retirement-loop-intro-modal'
        batches={[`retirement-1099R-4`]}
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
      />

      <DFModal
        i18nKey='/info/income/retirement/retirement-loop-intro-modal-mfj'
        batches={[`retirement-1099R-4`]}
        condition={{ condition: `/isFilingStatusMFJ` }}
      />

      <DFModal
        i18nKey='/info/income/retirement/supported-1099R-alert-modal'
        batches={[`retirement-1099R-4`]}
        condition={{ condition: `/has1099R` }}
      />

      <DFAlert
        condition={{ operator: `isFalseOrIncomplete`, condition: `/has1099R` }}
        i18nKey='/info/income/retirement/supported-1099R-alert'
        headingLevel='h2'
        batches={[`retirement-1099R-4`]}
        type='info'
      >
        <DFModal i18nKey='/info/income/retirement/supported-1099R-alert-modal' batches={[`retirement-1099R-4`]} />
      </DFAlert>

      <DFModal i18nKey='/info/income/retirement/explain-RRB-1099R-modal' batches={[`retirement-1099R-4`]} />

      <CollectionItemManager
        path='/form1099Rs'
        loopName='/form1099Rs'
        donePath='/hasCompleted1099RSection'
        batches={[`retirement-1099R-4`]}
      />
    </Screen>
    <CollectionLoop
      loopName='/form1099Rs'
      collection='/form1099Rs'
      collectionItemCompletedCondition='/form1099Rs/*/isComplete'
      donePath='/hasCompleted1099RSection'
      dataViewSections={[
        {
          i18nKey: `dataviews./flow/income/1099R.has1099R`,
          condition: `/form1099Rs/*/filer/isPrimaryFiler`,
        },
        {
          i18nKey: `dataviews./flow/income/1099R.secondaryFiler1099R`,
          condition: `/form1099Rs/*/belongsToSecondaryFiler`,
        },
      ]}
    >
      <SubSubcategory route='1099-r-information'>
        <Screen route='1099-r-nonstandard'>
          <Heading i18nKey='/heading/income/retirement/nonstandard-corrected' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/income/retirement/nonstandard-corrected' batches={[`retirement-1099R-1`]} />
          <Enum path='/form1099Rs/*/nonstandardOrCorrectedChoice' batches={[`retirement-1099R-3`]} />

          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-add-whose-1099-r' condition='/isFilingStatusMFJ'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-whose-retirement' batches={[`retirement-1099R-1`]} />
          <TaxReturnAlert
            i18nKey='/info/income/1099R/secondary-filer-income-without-mfj'
            headingLevel='h3'
            type='error'
            condition='/form1099Rs/*/secondaryFilerUsedWithoutMFJ'
          />
          <CollectionItemReference path='/form1099Rs/*/filer' displayOnlyOn='edit' batches={[`retirement-1099R-1`]} />
          <GenericString
            path='/form1099Rs/*/filer/firstName'
            displayOnlyOn='data-view'
            batches={[`retirement-1099R-1`]}
          />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='1099-r-add-recipient-name'>
          <Heading i18nKey='/heading/income/1099R/1099-r-add-recipient-name' batches={[`retirement-1099R-1`]} />
          <DFModal
            i18nKey='/info/income/1099R/why-change-name'
            condition='/form1099Rs/*/belongsToPrimaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <DFModal
            i18nKey='/info/income/1099R/why-change-name-spouse'
            condition='/form1099Rs/*/belongsToSecondaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/firstName'
            readOnly={true}
            displayOnlyOn='edit'
            condition='/form1099Rs/*/belongsToPrimaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/firstName'
            readOnly={true}
            displayOnlyOn='edit'
            condition='/form1099Rs/*/belongsToSecondaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/middleInitial'
            readOnly={true}
            displayOnlyOn='edit'
            condition='/form1099Rs/*/belongsToPrimaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/middleInitial'
            readOnly={true}
            displayOnlyOn='edit'
            condition='/form1099Rs/*/belongsToSecondaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/lastName'
            readOnly={true}
            displayOnlyOn='edit'
            condition='/form1099Rs/*/belongsToPrimaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/lastName'
            readOnly={true}
            displayOnlyOn='edit'
            condition='/form1099Rs/*/belongsToSecondaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <LimitingString
            path='/form1099Rs/*/filer/fullName'
            displayOnlyOn='data-view'
            batches={[`retirement-1099R-3`]}
          />
          <DFModal
            i18nKey='/info/income/1099R/incorrect-name'
            condition='/form1099Rs/*/belongsToPrimaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <DFModal
            i18nKey='/info/income/1099R/incorrect-name-spouse'
            condition='/form1099Rs/*/belongsToSecondaryFiler'
            batches={[`retirement-1099R-1`]}
          />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='1099-r-add-recipient-tin'>
          <Heading
            i18nKey='/heading/income/1099R/1099-r-add-recipient-tin-taxpayer'
            batches={[`retirement-1099R-1`]}
            condition='/form1099Rs/*/belongsToPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/1099R/1099-r-add-recipient-tin-spouse'
            batches={[`retirement-1099R-1`]}
            condition='/form1099Rs/*/belongsToSecondaryFiler'
          />
          <SetFactAction
            path='/form1099Rs/*/filer'
            source='/primaryFiler'
            conditions={[
              { operator: `isIncomplete`, condition: `/form1099Rs/*/filer` },
              { operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` },
            ]}
          />
          <Tin
            path='/form1099Rs/*/filer/tin'
            readOnly={true}
            hintKey='/info/income/1099R/why-change-tin'
            condition='/form1099Rs/*/belongsToPrimaryFiler'
            isSensitive={true}
            batches={[`retirement-1099R-1`]}
          />
          <Tin
            path='/form1099Rs/*/filer/tin'
            readOnly={true}
            hintKey='/info/income/1099R/why-change-tin-spouse'
            condition='/form1099Rs/*/belongsToSecondaryFiler'
            isSensitive={true}
            batches={[`retirement-1099R-1`]}
          />
          <DFModal i18nKey='/info/income/1099R/incorrect-tin' batches={[`retirement-1099R-1`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='1099-r-recipient-address-choice'>
          <Heading i18nKey='/heading/income/1099R/recipient-address-choice' batches={[`retirement-1099R-1`]} />
          <DFModal i18nKey='/info/retirement/1099r/what-if-1099r-new-address' batches={[`retirement-1099R-1`]} />
          <Enum path='/form1099Rs/*/recipientAddressChoice' batches={[`retirement-1099R-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='1099-r-add-recipient-address'
          condition={{ operator: `isTrue`, condition: `/form1099Rs/*/hasDifferentRecipientAddress` }}
        >
          <Heading i18nKey='/heading/income/1099R/recipient-different-address' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/retirement/1099r/recipient-different-address' batches={[`retirement-1099R-1`]} />
          <DFModal i18nKey='/info/retirement/1099r/what-if-1099r-new-address' batches={[`retirement-1099R-1`]} />
          <Address path='/form1099Rs/*/addressOverride' hintKey='/info/retirement/1099r/why-cant-i-change-country' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-payer-information'>
        <Screen route='1099-r-add-payer-info'>
          <Heading i18nKey='/heading/income/retirement/payer-info' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/retirement/1099r/what-is-payer' batches={[`retirement-1099R-1`]} />
          <LimitingString path='/form1099Rs/*/payer' batches={[`retirement-1099R-1`]} />
          <LimitingString
            path='/form1099Rs/*/writablePayerNameLine2'
            required={false}
            batches={[`retirement-1099R-1`]}
          />
          <Address
            path='/form1099Rs/*/payer/address'
            hintKey='/info/retirement/1099r/why-cant-i-change-country'
            batches={[`retirement-1099R-1`]}
          />
          <PhoneNumber required={false} path='/form1099Rs/*/payer/writablePhone' batches={[`retirement-1099R-1`]} />
          <Ein path='/form1099Rs/*/payer/tin' batches={[`retirement-1099R-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-distribution'>
        <Screen route='1099-r-add-box-1'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-1' batches={[`retirement-1099R-2`]} />
          <TaxReturnAlert
            i18nKey='/info/income/1099R/more-tax-than-distributions-alert'
            headingLevel='h3'
            type='error'
            condition='/form1099Rs/*/hasMoreTaxWithheldThanDistributions'
          />
          <DFModal i18nKey='/info/income/1099R/whats-gross-distribution' batches={[`retirement-1099R-2`]} />
          <Dollar path='/form1099Rs/*/writableGrossDistribution' batches={[`retirement-1099R-2`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-add-box-2a'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-2a' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-2a' batches={[`retirement-1099R-2`]} />
          <DFModal i18nKey='/info/income/1099R/whats-taxable-amount-mean' batches={[`retirement-1099R-3`]} />
          <DFModal i18nKey='/info/income/1099R/box-2a-enter-different' />
          <Dollar path='/form1099Rs/*/writableTaxableAmount' batches={[`retirement-1099R-3`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-add-box-2b'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-2b' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-2b' batches={[`retirement-1099R-2`]} />
          <Boolean
            path='/form1099Rs/*/writableTaxableAmountNotDetermined'
            batches={[`retirement-1099R-2`]}
            inputType='checkbox'
            required={false}
          />
          <Boolean
            path='/form1099Rs/*/writableTotalDistribution'
            batches={[`retirement-1099R-2`]}
            inputType='checkbox'
            required={false}
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='1099-r-box-2b-ko-1'
          condition='/flowKnockoutRetirementTaxableAmountNotDetermined'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-2b-ko-1' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099-r-box-2b-ko-1' batches={[`retirement-1099R-2`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen
          route='1099-r-box-2b-ko-2'
          condition='/flowKnockoutRetirementTaxableAmountNotDeterminedNotCheckedTaxableAmountBlank'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-2b-ko-2' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099-r-box-2b-ko-2' batches={[`retirement-1099R-2`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='1099-r-box-2b-ko-3' condition='/flowKnockoutRetirementTotalDistribution' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-2b-ko-3' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099-r-box-2b-ko-3' batches={[`retirement-1099R-2`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='1099-r-add-box-3'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-3' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-3' batches={[`retirement-1099R-2`]} />
          <Dollar path='/form1099Rs/*/writableCapitalGain' batches={[`retirement-1099R-2`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-3-ko' condition='/flowKnockoutRetirementCapitalGain' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-3' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099-r-box-3-ko' batches={[`retirement-1099R-2`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-federal-income-tax'>
        <Screen route='1099-r-add-box-4'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-4' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-4' batches={[`retirement-1099R-2`]} />
          <Dollar path='/form1099Rs/*/writableFederalWithholding' batches={[`retirement-1099R-2`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-contributions'>
        <Screen route='1099-r-add-box-5'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-5' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-5' batches={[`retirement-1099R-2`]} />
          <Dollar
            path='/form1099Rs/*/writableEmployeeOrRothOrInsuranceContributions'
            batches={[`retirement-1099R-2`]}
            required={false}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-net-appreciation'>
        <Screen route='1099-r-add-box-6'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-6' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-6' batches={[`retirement-1099R-2`]} />
          <DFModal i18nKey='/info/income/1099R/1099R-what-is-box-6' batches={[`retirement-1099R-2`]} />
          <Dollar path='/form1099Rs/*/writableNetAppreciation' batches={[`retirement-1099R-2`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-6-ko' condition='/flowKnockoutRetirementNetUnrealizedAppreciation' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-6' batches={[`retirement-1099R-2`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-6-ko' batches={[`retirement-1099R-2`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-distribution-codes'>
        <Screen route='retirement-add-box-7-checkbox'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-7-checkbox' batches={[`retirement-1099R-0`]} />
          <DFModal i18nKey='/info/income/1099R/1099R-what-is-box-7-checkbox' />
          <Boolean path='/form1099Rs/*/iraSepSimple' batches={[`retirement-1099R-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-7-checkbox-knockout' condition='/knockoutIraSepSimple' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading
            i18nKey='/heading/knockout/forms-missing/1099-r-unsupported-checkbox-IRA-SEP-Simple'
            batches={[`retirement-1099R-2`]}
          />
          <InfoDisplay
            i18nKey='/info/knockout/income/retirement/1099R-unsupported-checkbox-IRA-SEP-Simple'
            batches={[`retirement-1099R-2`]}
            condition='/form1099Rs/*/belongsToPrimaryFiler'
          />
          <InfoDisplay
            i18nKey='/info/knockout/income/retirement/1099R-unsupported-checkbox-IRA-SEP-Simple-Spouse'
            batches={[`retirement-1099R-2`]}
            condition='/form1099Rs/*/belongsToSecondaryFiler'
          />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='1099-r-add-box-7-codes'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-7' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-7' batches={[`retirement-1099R-1`]} />
          <DFModal i18nKey='/info/income/1099R/1099R-what-is-box-7' batches={[`retirement-1099R-1`]} />
          <LimitingString path='/form1099Rs/*/writableDistributionCode' batches={[`retirement-1099R-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-7-code-knockout' condition='/knockoutForForm1099RDistributionCode' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading
            i18nKey='/heading/knockout/forms-missing/1099-r-unsupported-distribution-code'
            batches={[`retirement-1099R-1`]}
          />
          <InfoDisplay
            i18nKey='/info/knockout/income/retirement/1099R-unsupported-distribution-code'
            batches={[`retirement-1099R-2`]}
          />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen
          route='1099-r-box-7-pretax-to-posttax-knockout'
          condition='/flowKnockoutForPreTaxToPostTaxRollover'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading
            i18nKey='/heading/knockout/forms-missing/1099-r-box-7-pretax-to-posttax-knockout'
            batches={[`retirement-1099R-0`]}
          />
          <InfoDisplay
            i18nKey='/info/knockout/income/retirement/1099-r-box-7-pretax-to-posttax-knockout'
            batches={[`retirement-1099R-0`]}
          />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Gate condition='/form1099Rs/*/flowQualifiedEarlyDistribution'>
          <Screen route='1099-r-box-7-early-distribution-question'>
            <Heading
              i18nKey='/heading/income/1099R/1099R-box-7-early-distribution'
              batches={[`retirement-1099R-0`]}
              condition='/form1099Rs/*/belongsToPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/income/1099R/1099R-box-7-early-distribution-spouse'
              batches={[`retirement-1099R-0`]}
              condition='/form1099Rs/*/belongsToSecondaryFiler'
            />
            <DFModal i18nKey='/info/income/1099R/whats-is-early-distribution' batches={[`retirement-1099R-0`]} />
            <Boolean
              path='/form1099Rs/*/writableQualifiedEarlyDistribution'
              batches={[`retirement-1099R-0`]}
              condition='/form1099Rs/*/belongsToPrimaryFiler'
              i18nKeySuffixContext='self'
            />
            <Boolean
              path='/form1099Rs/*/writableQualifiedEarlyDistribution'
              batches={[`retirement-1099R-0`]}
              condition='/form1099Rs/*/belongsToSecondaryFiler'
              i18nKeySuffixContext='mfj'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='1099-r-box-7-early-distribution-knockout'
            condition='/flowKnockoutRetirementQualifiedEarlyDistribution'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              i18nKey='/heading/knockout/rollovers/1099-r-early-distribution-knockout'
              batches={[`retirement-1099R-0`]}
            />
            <InfoDisplay
              i18nKey='/info/income/1099R/1099-r-early-distribution-knockout'
              batches={[`retirement-1099R-0`]}
            />
            <DFAlert
              i18nKey='/info/knockout/generic-other-ways-to-file'
              headingLevel='h2'
              type='warning'
              batches={[`retirement-1099R-0`]}
            />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </Gate>
        <Gate condition='/form1099Rs/*/flowAskAboutIndirectRollover'>
          <Screen route='1099-r-box-7-indirect-rollover'>
            <Heading
              i18nKey='/heading/income/1099R/1099R-box-7-indirect-rollover'
              batches={[`retirement-1099R-1`]}
              condition='/form1099Rs/*/belongsToPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/income/1099R/1099R-box-7-indirect-rollover-spouse'
              batches={[`retirement-1099R-1`]}
              condition='/form1099Rs/*/belongsToSecondaryFiler'
            />
            <InfoDisplay i18nKey='/info/income/1099R/1099R-box-7-indirect-rollover' batches={[`retirement-1099R-2`]} />
            <DFModal i18nKey='/info/income/1099R/whats-is-rolling-over-distribution' batches={[`retirement-1099R-2`]} />
            <Boolean
              condition='/form1099Rs/*/belongsToPrimaryFiler'
              i18nKeySuffixContext='self'
              path='/form1099Rs/*/writableIsIndirectRollover'
              batches={[`retirement-1099R-1`]}
            />

            <Boolean
              condition='/form1099Rs/*/belongsToSecondaryFiler'
              i18nKeySuffixContext='spouse'
              path='/form1099Rs/*/writableIsIndirectRollover'
              batches={[`retirement-1099R-1`]}
            />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='1099-r-box-7-indirect-rollover-knockout'
            condition='/flowKnockoutForRollover'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              i18nKey='/heading/knockout/rollovers/1099-r-indirect-distribution-knockout'
              batches={[`retirement-1099R-1`]}
            />
            <InfoDisplay
              i18nKey='/info/income/1099R/1099-r-indirect-distribution-knockout'
              batches={[`retirement-1099R-2`]}
            />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>

          <Screen route='1099-r-military-plan'>
            <Heading i18nKey='/heading/income/1099R/1099-r-military-plan' batches={[`retirement-1099R-3`]} />
            <InfoDisplay i18nKey='/info/income/1099R/1099-r-military-plan' batches={[`retirement-1099R-3`]} />
            <DFModal
              i18nKey='/info/income/1099R/what-is-distribution-from-military-plan'
              batches={[`retirement-1099R-3`]}
            />
            <Boolean
              path='/form1099Rs/*/writableIsDistributionFromMilitaryRetirementPlan'
              batches={[`retirement-1099R-3`]}
            />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen route='1099-r-disaster'>
            <Heading i18nKey='/heading/income/1099R/1099R-add-box-7-disaster' batches={[`retirement-1099R-0`]} />
            <DFModal
              i18nKey='/info/income/1099R/1099R-what-a-qualified-disaster-dist'
              batches={[`retirement-1099R-1`]}
            />
            <Boolean
              path='/form1099Rs/*/writableQualifiedDisasterDistribution'
              batches={[`retirement-1099R-1`]}
              condition='/form1099Rs/*/belongsToPrimaryFiler'
              i18nKeySuffixContext='self'
            />
            <Boolean
              path='/form1099Rs/*/writableQualifiedDisasterDistribution'
              batches={[`retirement-1099R-1`]}
              condition='/form1099Rs/*/belongsToSecondaryFiler'
              i18nKeySuffixContext='mfj'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='1099-r-disaster-knockout'
            condition='/flowKnockoutRetirementQualifiedDisasterDistribution'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              i18nKey='/heading/knockout/forms-missing/1099-r-qualified-disaster'
              batches={[`retirement-1099R-1`]}
            />
            <InfoDisplay
              i18nKey='/info/knockout/retirement/1099-r-qualified-disaster-knockout'
              batches={[`retirement-1099R-1`]}
            />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </Gate>
        <Gate condition='/form1099Rs/*/flowAskAboutPublicSafetyOfficer'>
          <Screen route='1099-r-pso'>
            <Heading
              i18nKey='/heading/income/1099R/1099-r-pso'
              batches={[`retirement-1099R-1`]}
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/income/1099R/1099-r-pso'
              batches={[`retirement-1099R-1`]}
              conditions={[
                { condition: `/form1099Rs/*/filer/isPrimaryFiler` },
                { condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/form1099Rs/*/filer/isSecondaryFiler` },
              ]}
            />
            <Heading
              i18nKey='/heading/income/1099R/1099-r-pso-spouse'
              batches={[`retirement-1099R-1`]}
              conditions={[
                { condition: `/form1099Rs/*/filer/isSecondaryFiler` },
                { condition: `/isFilingStatusMFJ` },
                { operator: `isFalse`, condition: `/form1099Rs/*/filer/isPrimaryFiler` },
              ]}
            />
            <DFModal i18nKey='/info/income/1099R/1099R-what-ask-retired' batches={[`retirement-1099R-1`]} />
            <Boolean
              path='/form1099Rs/*/writeablePublicSafetyOfficer'
              batches={[`retirement-1099R-1`]}
              condition='/form1099Rs/*/belongsToPrimaryFiler'
              i18nKeySuffixContext='self'
            />
            <Boolean
              path='/form1099Rs/*/writeablePublicSafetyOfficer'
              batches={[`retirement-1099R-1`]}
              condition='/form1099Rs/*/belongsToSecondaryFiler'
              i18nKeySuffixContext='mfj'
            />
            <SaveAndOrContinueButton />
          </Screen>

          <Gate condition='/form1099Rs/*/flowAskAboutPSOPremiums'>
            <Screen route='1099-r-pso-election'>
              <Heading
                i18nKey='/heading/income/1099R/1099-r-pso-premiums'
                batches={[`retirement-1099R-1`]}
                condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
              />
              <Heading
                i18nKey='/heading/income/1099R/1099-r-pso-premiums'
                batches={[`retirement-1099R-1`]}
                conditions={[
                  { condition: `/form1099Rs/*/filer/isPrimaryFiler` },
                  { condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/form1099Rs/*/filer/isSecondaryFiler` },
                ]}
              />
              <Heading
                i18nKey='/heading/income/1099R/1099-r-pso-premiums-spouse'
                batches={[`retirement-1099R-1`]}
                conditions={[
                  { condition: `/form1099Rs/*/filer/isSecondaryFiler` },
                  { condition: `/isFilingStatusMFJ` },
                  { operator: `isFalse`, condition: `/form1099Rs/*/filer/isPrimaryFiler` },
                ]}
              />
              <InfoDisplay i18nKey='/info/income/1099R/1099R-pso-premiums' batches={[`retirement-1099R-1`]} />
              <Boolean path='/form1099Rs/*/writeablePublicSafetyOfficerPremiums' batches={[`retirement-1099R-1`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='1099-r-pso-knockout'
              condition='/flowKnockoutRetirementPublicSafetyOfficer'
              isKnockout={true}
            >
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading
                i18nKey='/heading/knockout/forms-missing/1099-r-unsupported-public-safety'
                batches={[`retirement-1099R-1`]}
              />
              <InfoDisplay
                i18nKey='/info/knockout/1099-r-unsupported-public-safety-officer'
                batches={[`retirement-1099R-2`]}
              />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
          </Gate>
        </Gate>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-other'>
        <Screen route='1099-r-add-box-8'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-8' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-8' batches={[`retirement-1099R-1`]} />
          <Dollar path='/form1099Rs/*/writableOtherDollar' batches={[`retirement-1099R-1`]} required={false} />
          <LimitingString
            path='/form1099Rs/*/writableOtherPercentage'
            batches={[`retirement-1099R-2`]}
            required={false}
            inputSuffix='%'
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-8-value-knockout' condition='/knockoutForForm1099RBox8Value' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-8' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/knockout/1099R-box-8-knockout' batches={[`retirement-1099R-3`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen
          route='1099-r-box-8-percentage-knockout'
          condition='/knockoutForForm1099RBox8Percentage'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-8' batches={[`retirement-1099R-1`]} />
          <InfoDisplay i18nKey='/info/knockout/1099R-box-8-knockout' batches={[`retirement-1099R-3`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-total-distribution'>
        <Screen route='1099-r-add-box-9a'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-9a' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-9a' batches={[`retirement-1099R-3`]} />
          <LimitingString
            path='/form1099Rs/*/writablePercentageTotalDistribution'
            batches={[`retirement-1099R-3`]}
            required={false}
            inputSuffix='%'
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='1099-r-box-9a-percentage-knockout'
          condition='/knockoutForForm1099RBox9aPercentage'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-9a' />
          <InfoDisplay i18nKey='/info/knockout/1099R-box-9a-knockout' />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='1099-r-add-box-9b'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-9b' batches={[`retirement-1099R-0`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-9b' batches={[`retirement-1099R-0`]} />
          <Dollar
            path='/form1099Rs/*/writableTotalEmployeeContributions'
            batches={[`retirement-1099R-0`]}
            required={false}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-irr-allocable'>
        <Screen route='1099-r-add-box-10'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-10' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-10' batches={[`retirement-1099R-3`]} />
          <DFModal i18nKey='/info/income/1099R/whats-an-irr' batches={[`retirement-1099R-3`]} />
          <Dollar path='/form1099Rs/*/writableAmountAllocableToIRR' batches={[`retirement-1099R-3`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='1099-r-box-10-knockout'
          condition='/flowKnockoutRetirementAmountAllocableToIRR'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-10' batches={[`retirement-1099R-0`]} />
          <InfoDisplay i18nKey='/info/knockout/retirement/1099-r-allocable-irr' batches={[`retirement-1099R-0`]} />
          <DFAlert
            i18nKey='/info/knockout/generic-other-ways-to-file'
            headingLevel='h2'
            type='warning'
            batches={[`retirement-1099R-0`]}
          />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-roth-contributions'>
        <Screen route='1099-r-add-box-11'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-11' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-11' batches={[`retirement-1099R-3`]} />
          <LimitingString
            path='/form1099Rs/*/writableFirstYearDesignatedRothContributions'
            batches={[`retirement-1099R-3`]}
            required={false}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-facta'>
        <Screen route='1099-r-add-box-12'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-12-checkbox' batches={[`retirement-1099R-3`]} />
          <Boolean path='/form1099Rs/*/factaFilingRequirement' batches={[`retirement-1099R-3`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='1099-r-box-12-knockout'
          condition='/flowKnockoutRetirementFactaFilingRequirement'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-12' batches={[`retirement-1099R-0`]} />
          <InfoDisplay i18nKey='/info/knockout/retirement/1099-r-fatca-knockout' batches={[`retirement-1099R-0`]} />
          <DFAlert
            i18nKey='/info/knockout/generic-other-ways-to-file'
            headingLevel='h2'
            type='warning'
            batches={[`retirement-1099R-0`]}
          />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-13-date-of-payment'>
        <Screen route='1099-r-add-box-13'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-13' batches={[`retirement-1099R-0`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-13' batches={[`retirement-1099R-0`]} />
          <DatePicker path='/form1099Rs/*/writableDateOfPayment' required={false} batches={[`retirement-1099R-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-13-ko' condition='/flowKnockoutForDeathBenefitPaymentDate' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/1099-r-box-13' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099-r-box-13-ko' batches={[`retirement-1099R-3`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='1099-r-box-state-and-local'>
        <Screen route='1099-r-add-box-14'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-14' batches={[`retirement-1099R-3`]} />
          <TaxReturnAlert
            i18nKey='/info/income/1099R/more-tax-than-distributions-alert'
            headingLevel='h3'
            type='error'
            condition='/form1099Rs/*/hasMoreTaxWithheldThanDistributions'
          />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-14' batches={[`retirement-1099R-3`]} />
          <DFModal i18nKey='/info/income/1099R/what-if-2-amounts-box-14' />
          <Dollar path='/form1099Rs/*/writableStateTaxWithheld' batches={[`retirement-1099R-3`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-check-state-value' condition='/form1099Rs/*/missingStateTaxWithheld'>
          <Heading i18nKey='/heading/income/1099R/1099R-check-state-value' />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-check-state-value' />
          <SaveAndOrContinueButton i18nKey='button.continueWoStateTaxWithheld' isOutline />
        </Screen>
        <Screen route='1099-r-add-box-15'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-15' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-15' batches={[`retirement-1099R-3`]} />
          <DFModal i18nKey='/info/income/1099R/what-if-2-amounts-box-15' batches={[`retirement-1099R-3`]} />
          <Enum
            path='/form1099Rs/*/writablePayerState'
            batches={[`retirement-1099R-3`]}
            required={false}
            renderAs='select'
          />
          <LimitingString
            path='/form1099Rs/*/writablePayerStateNumber'
            batches={[`retirement-1099R-3`]}
            required={false}
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='1099-r-box-15-ko' condition='/flowKnockoutBox15' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/income/1099R/1099R-add-box-15-ko' batches={[`retirement-1099R-3`]} />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-box-15-ko' batches={[`retirement-1099R-3`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='1099-r-add-boxes-16-19'>
          <Heading i18nKey='/heading/income/1099R/1099R-add-boxes-16-19' batches={[`retirement-1099R-0`]} />
          <TaxReturnAlert
            i18nKey='/info/income/1099R/more-tax-than-distributions-alert'
            headingLevel='h3'
            type='error'
            condition='/form1099Rs/*/hasMoreTaxWithheldThanDistributions'
            factPaths={[`/form1099Rs/*/writableLocalTaxWithheld`]}
          />
          <InfoDisplay i18nKey='/info/income/1099R/1099R-add-boxes-16-19' batches={[`retirement-1099R-0`]} />
          <DFModal i18nKey='/info/income/1099R/what-if-multiple-amounts-box-16-19' />
          <Dollar path='/form1099Rs/*/writableStateDistribution' batches={[`retirement-1099R-0`]} required={false} />
          <Dollar path='/form1099Rs/*/writableLocalTaxWithheld' batches={[`retirement-1099R-0`]} required={false} />
          <LimitingString path='/form1099Rs/*/writableLocality' batches={[`retirement-1099R-0`]} required={false} />
          <Dollar path='/form1099Rs/*/writableLocalDistribution' batches={[`retirement-1099R-0`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/form1099Rs/*/hasMoreTaxWithheldThanDistributions'>
          <Screen route='1099-r-state-local-tax-error' condition='/hasMoreTaxThanDistributionsOn1099Rs'>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/income/1099R/more-tax-than-distributions' />
            <DFAlert i18nKey='/info/income/1099R/more-tax-than-distributions' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </Gate>
      </SubSubcategory>
      <SubSubcategory route='1099-r-account-number'>
        <Screen route='1099-r-add-account-number'>
          <Heading i18nKey='/heading/income/retirement/account-number' batches={[`retirement-1099R-0`]} />
          <InfoDisplay i18nKey='/info/income/retirement/account-number' batches={[`retirement-1099R-0`]} />
          <LimitingString
            path='/form1099Rs/*/writableAccountNumber'
            batches={[`retirement-1099R-0`]}
            required={false}
          />
          <SetFactAction path='/form1099Rs/*/hasSeenLastAvailableScreen' source='/flowTrue' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </CollectionLoop>
  </Subcategory>
);

export const PreReleaseRetirementSubcategory = (
  <Subcategory
    route='retirement-launch'
    completeIf='/form1099RsIsComplete'
    displayOnlyIf={{ operator: `isFalseOrIncomplete`, condition: `/is1099RFeatureFlagEnabled` }}
    dataItems={[
      {
        itemKey: `retirementTaxable`,
        conditions: [`/has1099R`],
      },
      {
        itemKey: `retirementNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/has1099R` }],
      },
    ]}
  >
    <SubSubcategory route='1099-r-info'>
      <Screen route='1099-r-launch'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/income/retirement/retirement-loop'
          batches={[`retirement-1099R-4`]}
          condition={{ operator: `isFalseOrIncomplete`, condition: `/has1099R` }}
        />
        <Heading
          i18nKey='/heading/income/retirement/retirement-loop'
          batches={[`retirement-1099R-4`]}
          condition='/has1099R'
        />

        <Heading
          i18nKey='/heading/income/retirement/retirement-loop-intro'
          batches={[`retirement-1099R-4`]}
          conditions={[
            { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
            { operator: `isFalseOrIncomplete`, condition: `/has1099R` },
          ]}
        />
        <Heading
          i18nKey='/heading/income/retirement/retirement-loop-intro-mfj'
          batches={[`retirement-1099R-4`]}
          conditions={[
            { condition: `/isFilingStatusMFJ` },
            { operator: `isFalseOrIncomplete`, condition: `/has1099R` },
          ]}
        />
        <DFModal
          i18nKey='/info/income/retirement/retirement-loop-intro-modal'
          batches={[`retirement-1099R-4`]}
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />

        <DFModal
          i18nKey='/info/income/retirement/retirement-loop-intro-modal-mfj'
          batches={[`retirement-1099R-4`]}
          condition={{ condition: `/isFilingStatusMFJ` }}
        />

        <DFModal
          i18nKey='/info/income/retirement/supported-1099R-alert-modal'
          batches={[`retirement-1099R-4`]}
          condition={{ condition: `/has1099R` }}
        />

        <DFAlert
          condition={{ operator: `isFalseOrIncomplete`, condition: `/has1099R` }}
          i18nKey='/info/income/retirement/launch-1099R-alert'
          headingLevel='h2'
          batches={[`retirement-1099R-4`]}
          type='info'
        >
          <DFModal i18nKey='/info/income/retirement/supported-1099R-alert-modal' batches={[`retirement-1099R-4`]} />
        </DFAlert>
        <DFModal i18nKey='/info/income/retirement/explain-RRB-1099R-modal' batches={[`retirement-1099R-4`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='1099-r-form-add'>
        <Heading
          i18nKey='/heading/income/retirement/retirement-had-1099r'
          batches={[`retirement-1099R-4`]}
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <Heading
          i18nKey='/heading/income/retirement/retirement-had-1099r-mfj'
          batches={[`retirement-1099R-4`]}
          condition='/isFilingStatusMFJ'
        />

        <Boolean path='/hasSome1099rFormsBeforeSectionEnabled' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='1099-r-add-form-ko' condition='/has1099rFormsAndFeatureIsNotEnabled' isKnockout={true}>
        <IconDisplay name='CalendarToday' size={9} isCentered />
        <Heading
          i18nKey='/heading/income/retirement/retirement-had-1099r-before-section-enabled'
          batches={[`retirement-1099R-4`]}
        />
        <DFModal
          i18nKey='/info/income/retirement/retirement-had-1099r-before-section-enabled'
          batches={[`retirement-1099R-4`]}
        />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>
  </Subcategory>
);
