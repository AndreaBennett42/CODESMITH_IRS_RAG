/* eslint-disable max-len */
import { Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ConditionalList,
  ContextHeading,
  DFAlert,
  DFModal,
  Dollar,
  Enum,
  Heading,
  IconDisplay,
  InfoDisplay,
  KnockoutButton,
  SaveAndOrContinueButton,
  Tin,
} from '../../ContentDeclarations.js';

export const EstimatedTaxesSubcategory = (
  <Subcategory route='estimated-taxes-paid' completeIf='/estimatedTaxesIsComplete'>
    <Screen route='est-tax-payments-intro'>
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/your-taxes/estimated-taxes-paid' />
      <Heading i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-intro' />
      <InfoDisplay i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-intro' />
      <ConditionalList
        i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-intro'
        items={[{ itemKey: `listItem_taxPayments` }, { itemKey: `listItem_overPayments` }]}
      />
      <SaveAndOrContinueButton />
    </Screen>
    <SubSubcategory route='est-tax-payments'>
      <Screen route='est-tax-payments-y-n'>
        <Heading
          i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-y-n'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <Heading
          i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-y-n-mfj'
          condition='/isFilingStatusMFJ'
        />
        <DFModal
          i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-yn'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <DFModal
          i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-yn-mfj'
          condition='/isFilingStatusMFJ'
        />
        <Boolean path='/paidEstimatedTaxesOrFromLastYear' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='no-payments-hsa-distributions-ko' condition='/knockoutNoPaymentsHsaDistribution' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/hsa-no-payments-distribution-ko' batches={[`hsa-1`]} />
        <InfoDisplay
          i18nKey='/info/knockout/hsa-no-payments-distribution-ko'
          batches={[`hsa-1`]}
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <InfoDisplay
          i18nKey='/info/knockout/hsa-no-payments-distribution-ko-mfj'
          batches={[`hsa-1`]}
          condition='/isFilingStatusMFJ'
        />
        <DFAlert i18nKey='/info/knockout/how-to-file-paper' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      <Screen route='no-payments-ko' condition='/knockoutHasNoPaymentsOrCredits' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/no-payments-or-credit' />
        <InfoDisplay i18nKey='/info/knockout/no-payments-or-credit' />
        <InfoDisplay i18nKey='/info/knockout/state-filing' condition='/mayBenefitFromFilingStateTaxes' />
        <DFAlert
          i18nKey='/info/knockout/dont-need-to-file'
          headingLevel='h2'
          type='warning'
          condition={{ operator: `isFalseOrIncomplete`, condition: `/mayBenefitFromFilingStateTaxes` }}
        />
        <DFAlert
          i18nKey='/info/knockout/dont-need-to-file-state'
          headingLevel='h2'
          type='warning'
          condition='/mayBenefitFromFilingStateTaxes'
        />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>

    <Gate condition='/paidEstimatedTaxesOrFromLastYear'>
      <SubSubcategory route='est-tax-payments'>
        <Screen route='est-tax-payments-name-change'>
          <Heading
            i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-name-change'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-name-change-mfj'
            condition='/isFilingStatusMFJ'
          />
          <Boolean path='/paidEstimatedTaxesOrFromLastYearUnderDifferentName' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='est-tax-payments-name-change-ko'
          condition='/knockoutPaidEstimatedTaxesUnderDifferentLastName'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/est-tax-payments-name-change' />
          <InfoDisplay
            i18nKey='/info/knockout/est-tax-payments-name-change-details'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <InfoDisplay
            i18nKey='/info/knockout/est-tax-payments-name-change-details-mfj'
            condition='/isFilingStatusMFJ'
          />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='est-tax-payments-divorced'>
          <Heading i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-divorced' />
          <Enum path='/paidEstimatedTaxesWithFormerSpouse' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='former-spouse-info'>
        <Screen
          route='est-tax-payments-divorced-tax-id-primary-filer'
          condition='/tpPaidEstimatedTaxesWithFormerSpouse'
        >
          <Heading i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-divorced-tax-id-primary-filer' />
          <InfoDisplay i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-divorced-tax-id-input-details' />
          <Tin path='/primaryFilerDivorcedSpouseTaxID' isSensitive={true} />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='former-spouse-info'>
        <Screen
          route='est-tax-payments-divorced-tax-id-secondary-filer'
          condition='/spousePaidEstimatedTaxesWithFormerSpouse'
        >
          <Heading i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-divorced-tax-id-secondary-filer' />
          <InfoDisplay i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-divorced-tax-id-input-details' />
          <Tin path='/secondaryFilerDivorcedSpouseTaxID' isSensitive={true} />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='amount-paid'>
        <Screen route='est-tax-payments-amount'>
          <Heading
            i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-amount'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/your-taxes/estimated-taxes-paid/est-tax-payments-amount-mfj'
            condition='/isFilingStatusMFJ'
          />
          <DFModal
            i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-amount-details'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <DFModal
            i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-amount-details-mfj'
            condition='/isFilingStatusMFJ'
          />
          <DFModal
            i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-amount-snack1'
            condition='/someTPPaidEstimatedTaxesWithFormerSpouse'
          />
          <DFModal
            i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-amount-snack2'
            condition='/isFilingStatusMFS'
          />
          <DFModal
            i18nKey='/info/your-taxes/estimated-taxes-paid/est-tax-payments-amount-snack3'
            condition='/isFilingStatusMFJ'
          />
          <Dollar path='/estimatedTaxPaymentWritable' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </Gate>
  </Subcategory>
);
