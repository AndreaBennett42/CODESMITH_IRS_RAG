/* eslint-disable max-len */
import { CollectionLoop, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  CollectionItemManager,
  CollectionItemReference,
  ContextHeading,
  DFModal,
  DFAlert,
  Dollar,
  Enum,
  Ein,
  GenericString,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
  SetFactAction,
  LimitingString,
  TaxReturnAlert,
  IconDisplay,
  KnockoutButton,
  Hint,
} from '../../ContentDeclarations.js';

export const UnemploymentIncomeSubcategory = (
  <Subcategory
    route='unemployment'
    completeIf='/form1099GsIsDone'
    collectionContext='/form1099Gs'
    dataItems={[
      {
        itemKey: `unemploymentTaxable`,
        conditions: [`/receivedUnemploymentCompensation`],
      },
      {
        itemKey: `unemploymentNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/receivedUnemploymentCompensation` }],
      },
    ]}
  >
    <Screen route='unemployment-loop-intro'>
      <ContextHeading
        displayOnlyOn='edit'
        i18nKey='/heading/income/unemployment/unemployment-loop'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasForm1099Gs` }}
      />
      <Heading i18nKey='/heading/income/unemployment/unemployment-loop' condition='/hasForm1099Gs' />
      <Heading
        i18nKey='/heading/income/unemployment/unemployment-loop-intro'
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasForm1099Gs` },
        ]}
      />
      <Heading
        i18nKey='/heading/income/unemployment/unemployment-loop-intro-mfj'
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasForm1099Gs` },
        ]}
      />
      <DFModal i18nKey='/info/income/unemployment/unemployment-loop-intro' />
      <DFAlert
        i18nKey='/info/income/unemployment/1099G-unsupported-income-types-alert'
        headingLevel='h3'
        type='info'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasForm1099Gs` }}
      >
        <DFModal i18nKey='/info/income/unemployment/1099G-unsupported-income-types-alert-modal' />
      </DFAlert>
      <DFModal i18nKey='/info/income/unemployment/1099G-unsupported-income-types' condition='/hasForm1099Gs' />
      <DFModal i18nKey='/info/income/unemployment/1099G-calculations' condition='/hasForm1099Gs' />

      <CollectionItemManager
        path='/form1099Gs'
        loopName='/form1099Gs'
        donePath='/form1099GsIsDone'
        batches={[`information-architecture-0`]}
      />
    </Screen>
    <CollectionLoop
      loopName='/form1099Gs'
      collection='/form1099Gs'
      iconName='AttachMoney'
      collectionItemCompletedCondition='/form1099Gs/*/isComplete'
      donePath='/form1099GsIsDone'
      dataViewSections={[
        {
          i18nKey: `dataviews./flow/income/unemployment.primaryFiler`,
          condition: `/form1099Gs/*/filer/isPrimaryFiler`,
        },
        {
          i18nKey: `dataviews./flow/income/unemployment.secondaryFiler`,
          condition: `/form1099Gs/*/filer/isSecondaryFiler`,
        },
      ]}
    >
      <SubSubcategory route='int-income-basic-info'>
        <Screen route='add-1099-g'>
          <Heading
            i18nKey='/heading/income/unemployment/add-1099'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading i18nKey='/heading/income/unemployment/add-1099-mfj' condition='/isFilingStatusMFJ' />
          <SetFactAction
            path='/form1099Gs/*/filer'
            source='/primaryFiler'
            condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
          />

          <Boolean path='/form1099Gs/*/has1099' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Gate condition='/form1099Gs/*/has1099'>
        <SubSubcategory route='int-income-basic-info'>
          <Screen route='1099-g-add-whose' condition='/isFilingStatusMFJ'>
            <TaxReturnAlert
              i18nKey='/info/income/unemployment/secondary-filer-income-without-mfj'
              headingLevel='h3'
              type='error'
              condition='/form1099Gs/*/secondaryFilerUsedWithoutMFJ'
            />
            <Heading i18nKey='/heading/income/unemployment/1099-g-add-whose' />
            <CollectionItemReference path='/form1099Gs/*/filer' displayOnlyOn='edit' />
            <GenericString path='/form1099Gs/*/filer/fullName' displayOnlyOn='data-view' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-g-payer'>
            <Heading i18nKey='/heading/income/unemployment/payer' />
            <InfoDisplay i18nKey='/info/income/unemployment/payer' />
            <LimitingString path='/form1099Gs/*/payer' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='compensation-amount'>
          <Screen route='1099-g-amount'>
            <Heading i18nKey='/heading/income/unemployment/1099-g-amount' />
            <Hint i18nKey='/income/unemployment/1099-g-amount' />
            <Dollar path='/form1099Gs/*/amount' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-g-repay-options'>
            <Heading
              i18nKey='/heading/income/unemployment/1099-g-repay-options'
              condition={{ operator: `isFalse`, condition: `/form1099Gs/*/isMfjAndSpouses1099G` }}
            />
            <Heading
              i18nKey='/heading/income/unemployment/1099-g-repay-options-spouse'
              condition='/form1099Gs/*/isMfjAndSpouses1099G'
            />
            <InfoDisplay i18nKey='/info/income/unemployment/1099-g-repay-options' />
            <Boolean path='/form1099Gs/*/repaid' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-g-repay-amount' condition='/form1099Gs/*/repaid'>
            <Heading i18nKey='/heading/income/unemployment/1099-g-repay-amount' />
            <Dollar path='/form1099Gs/*/writableAmountPaidBackForBenefitsInTaxYear' required={false} />
            <Dollar path='/form1099Gs/*/writableAmountPaidBackForBenefitsBeforeTaxYear' required={false} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-g-repay-knockout' condition='/flowKnockout1099GRepayTooHigh' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              i18nKey='/heading/knockout/income/unemployment/1099g-repay-ko'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/knockout/income/unemployment/1099g-repay-ko-mfj'
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay
              i18nKey='/info/knockout/income/unemployment/1099g-repay-ko'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <InfoDisplay
              i18nKey='/info/knockout/income/unemployment/1099g-repay-ko-mfj'
              condition='/isFilingStatusMFJ'
            />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='federal-income-tax-withheld'>
          <Screen route='1099-g-withholding'>
            <Heading i18nKey='/heading/income/unemployment/withholding' />
            <InfoDisplay i18nKey='/info/income/unemployment/withholding' />
            <Dollar path='/form1099Gs/*/writableFederalTaxWithheld' required={false} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='payer-tin'>
          <Screen route='1099-g-payer-tin' condition='/form1099Gs/*/flowHasFederalTaxWithheld'>
            <Heading i18nKey='/heading/income/unemployment/payer-tin' />
            <InfoDisplay i18nKey='/info/income/unemployment/payer-tin' />
            <Ein path='/form1099Gs/*/payer/tin' isSensitive={true} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='state-tax-information'>
          <Screen route='1099-g-state-info'>
            <Heading i18nKey='/heading/income/unemployment/state-info' />
            <InfoDisplay i18nKey='/info/income/unemployment/state-info' />
            <Enum path='/form1099Gs/*/writableState' required={false} renderAs='select' />
            <GenericString path='/form1099Gs/*/writableStateIdNumber' required={false} />
            <Dollar path='/form1099Gs/*/writableStateTaxWithheld' required={false} />
            <SetFactAction path='/form1099Gs/*/hasSeenLastAvailableScreen' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='1099-g-other-state-ko'
            condition='/flowKnockoutUnemploymentCompensationUnsupportedState'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/income/unemployment/1099g-other-state-ko' />
            <InfoDisplay i18nKey='/info/knockout/income/unemployment/1099g-other-state-ko' />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
      </Gate>
      <Gate condition={{ operator: `isFalse`, condition: `/form1099Gs/*/has1099` }}>
        <SubSubcategory route='int-income-basic-info'>
          <Screen route='no-1099-g-add-whose' condition='/isFilingStatusMFJ'>
            <Heading i18nKey='/heading/income/unemployment/no-1099-g-add-whose' />
            <CollectionItemReference path='/form1099Gs/*/filer' displayOnlyOn='edit' />
            <GenericString path='/form1099Gs/*/filer/fullName' displayOnlyOn='data-view' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='no-1099-g-payer-name'>
            <Heading i18nKey='/heading/income/unemployment/payer' />
            <InfoDisplay i18nKey='/info/income/unemployment/no-payer' />
            <LimitingString path='/form1099Gs/*/payer' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='compensation-amount'>
          <Screen route='no-1099-g-amount'>
            <Heading i18nKey='/heading/income/unemployment/no-1099-g-amount' />
            <Dollar path='/form1099Gs/*/amount' />
            <SetFactAction path='/form1099Gs/*/hasSeenLastAvailableScreen' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='no-1099-g-repay-options'>
            <Heading
              i18nKey='/heading/income/unemployment/1099-g-repay-options'
              condition={{ operator: `isFalse`, condition: `/form1099Gs/*/isMfjAndSpouses1099G` }}
            />
            <Heading
              i18nKey='/heading/income/unemployment/1099-g-repay-options-spouse'
              condition='/form1099Gs/*/isMfjAndSpouses1099G'
            />
            <InfoDisplay i18nKey='/info/income/unemployment/no-1099-g-repay-options' />
            <Boolean path='/form1099Gs/*/repaid' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='no-1099-g-repay-amount' condition='/form1099Gs/*/repaid'>
            <Heading i18nKey='/heading/income/unemployment/1099-g-repay-amount' />
            <Dollar path='/form1099Gs/*/writableAmountPaidBackForBenefitsInTaxYear' required={false} />
            <Dollar path='/form1099Gs/*/writableAmountPaidBackForBenefitsBeforeTaxYear' required={false} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='no-1099-g-repay-knockout' condition='/flowKnockout1099GRepayTooHigh' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              i18nKey='/heading/knockout/income/unemployment/1099g-repay-ko'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <Heading
              i18nKey='/heading/knockout/income/unemployment/1099g-repay-ko-mfj'
              condition='/isFilingStatusMFJ'
            />
            <InfoDisplay
              i18nKey='/info/knockout/income/unemployment/1099g-repay-ko'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <InfoDisplay
              i18nKey='/info/knockout/income/unemployment/1099g-repay-ko-mfj'
              condition='/isFilingStatusMFJ'
            />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
      </Gate>
    </CollectionLoop>
  </Subcategory>
);
