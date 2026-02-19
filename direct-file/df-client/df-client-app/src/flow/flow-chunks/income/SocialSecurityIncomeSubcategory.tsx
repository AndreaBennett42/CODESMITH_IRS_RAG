/* eslint-disable max-len */
import { CollectionLoop, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  CollectionItemManager,
  CollectionItemReference,
  ContextHeading,
  DFModal,
  DFAlert,
  Dollar,
  GenericString,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
  SetFactAction,
  TaxReturnAlert,
  IconDisplay,
  KnockoutButton,
} from '../../ContentDeclarations.js';

export const SocialSecurityIncomeSubcategory = (
  <Subcategory
    route='social-security'
    completeIf='/socialSecurityReportsIsDone'
    collectionContext='/socialSecurityReports'
    dataItems={[
      {
        itemKey: `socialSecurityTaxable`,
        conditions: [`/hasSocialSecurityBenefits`],
      },
      {
        itemKey: `socialSecurityNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/hasSocialSecurityBenefits` }],
      },
    ]}
  >
    <Screen route='social-security-loop-intro'>
      <ContextHeading
        displayOnlyOn='edit'
        i18nKey='/heading/income/other-income/social-security'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasSocialSecurityReports` }}
      />
      <Heading i18nKey='/heading/income/other-income/social-security' condition='/hasSocialSecurityReports' />
      <Heading
        i18nKey='/heading/income/other-income/social-security/ssa-or-rrb-info'
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasSocialSecurityReports` },
        ]}
      />
      <Heading
        i18nKey='/heading/income/other-income/social-security/ssa-or-rrb-info-mfj'
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasSocialSecurityReports` },
        ]}
      />
      <DFModal i18nKey='/info/income/other-income/social-security/ssa-or-rrb-info' batches={[`updates-0`]} />
      <DFAlert
        i18nKey={null}
        headingLevel='h3'
        type='info'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasSocialSecurityReports` }}
      >
        <DFModal i18nKey='/info/income/other-income/social-security/unsupported-social-security-benefits-alert-modal' />
      </DFAlert>
      <DFModal
        i18nKey='/info/income/other-income/social-security/unsupported-social-security-benefits'
        condition='/hasSocialSecurityReports'
      />
      <DFModal
        i18nKey='/info/income/other-income/social-security/social-security-calculations'
        conditions={[{ condition: `/hasSocialSecurityReports` }, { condition: `/socialSecurityBenefitsAreLossUpto3k` }]}
      />
      <DFModal
        i18nKey='/info/income/other-income/social-security/social-security-calculations-positive'
        conditions={[
          { condition: `/hasSocialSecurityReports` },
          { operator: `isFalse`, condition: `/socialSecurityBenefitsAreLossUpto3k` },
        ]}
        batches={[`updates-0`]}
      />
      <CollectionItemManager
        path='/socialSecurityReports'
        loopName='/socialSecurityReports'
        donePath='/socialSecurityReportsIsDone'
        batches={[`information-architecture-0`]}
      />
      <InfoDisplay
        i18nKey='/info/income/other-income/social-security/taxable-context'
        condition='/hasAtLeastOneSocialSecurityReport'
      />
    </Screen>
    <CollectionLoop
      loopName='/socialSecurityReports'
      collection='/socialSecurityReports'
      collectionItemCompletedCondition='/socialSecurityReports/*/isComplete'
      iconName='AttachMoney'
      donePath='/socialSecurityReportsIsDone'
      knockoutRoute='/flow/income/social-security/social-security-summary-negative-ko'
      dataViewSections={[
        {
          i18nKey: `dataviews./flow/income/social-security.primaryFiler`,
          condition: `/socialSecurityReports/*/filer/isPrimaryFiler`,
        },
        {
          i18nKey: `dataviews./flow/income/social-security.secondaryFiler`,
          condition: `/socialSecurityReports/*/filer/isSecondaryFiler`,
        },
      ]}
    >
      {/* <SubSubcategory route='benefit-income-basic-info'>
        <Screen route='ssa-or-rrb-choice'>
          <Heading i18nKey='/heading/income/other-income/social-security/ssa-or-rrb-choice' />
          <SetFactAction
            path='/socialSecurityReports/*\/filer'
            source='/primaryFiler'
            condition={{ operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` }}
          />
          <DFModal i18nKey='/info/income/other-income/social-security/1099-r' />
          <Enum path='/socialSecurityReports/*\/formType' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory> */}
      {/* <Gate condition='/socialSecurityReports/*\/isSsa1099'> */}
      <SubSubcategory route='benefit-income-basic-info'>
        <Screen route='ssa-whose' condition='/isFilingStatusMFJ'>
          <TaxReturnAlert
            i18nKey='/info/income/other-income/social-security/secondary-filer-income-without-mfj'
            headingLevel='h3'
            type='error'
            condition='/socialSecurityReports/*/secondaryFilerUsedWithoutMFJ'
          />
          <Heading i18nKey='/heading/income/other-income/social-security/add-whose-ss' />
          <CollectionItemReference path='/socialSecurityReports/*/filer' displayOnlyOn='edit' />
          <GenericString path='/socialSecurityReports/*/filer/fullName' displayOnlyOn='data-view' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='net-benefits'>
        <Screen route='ssa-1099-box-5'>
          <Heading i18nKey='/heading/income/other-income/social-security/ssa-1099-box-5' />
          <InfoDisplay i18nKey='/info/income/other-income/social-security/ssa-rrb-1099-box-5' />
          <Dollar path='/socialSecurityReports/*/ssaNetBenefits' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Screen
        route='social-security-summary-negative-ko-in-flow'
        condition='/socialSecurityBenefitsAreLossGreaterThan3k'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/social-security-summary-negative' />
        <InfoDisplay
          i18nKey='/info/knockout/social-security-summary-negative'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <InfoDisplay i18nKey='/info/knockout/social-security-summary-negative-mfj' condition='/isFilingStatusMFJ' />
        <DFAlert i18nKey='/info/knockout/social-security-summary-negative' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      <SubSubcategory route='federal-income-taxes-withheld'>
        <Screen route='ssa-1099-box-6'>
          <Heading i18nKey='/heading/income/other-income/social-security/ssa-1099-box-6' />
          <InfoDisplay i18nKey='/info/income/other-income/social-security/ssa-1099-box-6' />
          <Dollar path='/socialSecurityReports/*/writableSsaFederalTaxWithheld' required={false} />
          <SetFactAction path='/socialSecurityReports/*/hasSeenLastAvailableScreen' source='/flowTrue' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      {/* </Gate> */}
      <Gate condition={{ operator: `isFalse`, condition: `/socialSecurityReports/*/isSsa1099` }}>
        <SubSubcategory route='benefit-income-basic-info'>
          <Screen route='rrb-whose' condition='/isFilingStatusMFJ'>
            <Heading i18nKey='/heading/income/other-income/social-security/add-whose-rrb' />
            <CollectionItemReference path='/socialSecurityReports/*/filer' displayOnlyOn='edit' />
            <GenericString path='/socialSecurityReports/*/filer/fullName' displayOnlyOn='data-view' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='net-benefits'>
          <Screen route='rrb-1099-box-5'>
            <Heading i18nKey='/heading/income/other-income/social-security/rrb-1099-box-5' />
            <InfoDisplay i18nKey='/info/income/other-income/social-security/ssa-rrb-1099-box-5' />
            <Dollar path='/socialSecurityReports/*/rrbNetBenefits' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='federal-income-taxes-withheld'>
          <Screen route='rrb-1099-box-10'>
            <Heading i18nKey='/heading/income/other-income/social-security/rrb-1099-box-10' />
            <InfoDisplay i18nKey='/info/income/other-income/social-security/rrb-1099-box-10' />
            <Dollar path='/socialSecurityReports/*/writableRrbFederalTaxWithheld' required={false} />
            <SetFactAction path='/socialSecurityReports/*/hasSeenLastAvailableScreen' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Gate>
    </CollectionLoop>
    <Screen
      route='social-security-summary-negative-ko'
      condition='/socialSecurityBenefitsAreLossGreaterThan3k'
      isKnockout={true}
    >
      <IconDisplay name='ErrorOutline' size={9} isCentered />
      <Heading i18nKey='/heading/knockout/social-security-summary-negative' />
      <InfoDisplay
        i18nKey='/info/knockout/social-security-summary-negative'
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
      />
      <InfoDisplay i18nKey='/info/knockout/social-security-summary-negative-mfj' condition='/isFilingStatusMFJ' />
      <DFAlert i18nKey='/info/knockout/social-security-summary-negative' headingLevel='h2' type='warning' />
      <KnockoutButton i18nKey='button.knockout' />
    </Screen>
  </Subcategory>
);
