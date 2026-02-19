/* eslint-disable max-len */
import { CollectionLoop, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  CollectionItemManager,
  CollectionItemReference,
  ConditionalList,
  ContextHeading,
  DFModal,
  DFAlert,
  Dollar,
  Enum,
  Ein,
  GenericString,
  Heading,
  InfoDisplay,
  IntroContent,
  LimitingString,
  SaveAndOrContinueButton,
  SetFactAction,
  TaxReturnAlert,
  IconDisplay,
  KnockoutButton,
  CollectionDataPreview,
  DFAccordion,
} from '../../ContentDeclarations.js';

export const InterestIncomeSubcategory = (
  <Subcategory
    route='interest'
    completeIf='/interestReportsIsDone'
    collectionContext='/interestReports'
    dataItems={[
      {
        itemKey: `interestTaxable`,
        conditions: [`/hasInterestIncome`],
      },
      {
        itemKey: `interestNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/hasInterestIncome` }],
      },
    ]}
  >
    <Gate condition={{ condition: `data-import`, section: `1099-ints` }}>
      <Screen route='int-income-loop-intro-data-import'>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/income/interest'
          condition={{ operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` }}
        />
        <Heading
          i18nKey='/heading/income/interest/intro'
          conditions={[
            { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
            { operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` },
          ]}
          batches={[`schedule-b-0`]}
        />
        <Heading
          i18nKey='/heading/income/interest/intro-mfj'
          conditions={[
            { condition: `/isFilingStatusMFJ` },
            { operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` },
          ]}
          batches={[`schedule-b-0`]}
        />
        <IntroContent i18nKey='/info/income/interest/intro-data-import' />
        <ConditionalList
          i18nKey='/info/income/interest/intro-list-data-import'
          items={[
            {
              itemKey: `listItem_any_1099_forms`,
              conditions: [{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }],
            },
            { itemKey: `listItem_any_1099_forms-primary-filer`, conditions: [`/isFilingStatusMFJ`] },
            { itemKey: `listItem_any_1099_forms-secondary-filer`, conditions: [`/isFilingStatusMFJ`] },
            { itemKey: `listItem_information_taxable_interest` },
          ]}
        />
        <DFAlert
          i18nKey='/info/income/interest/unsupported-interest-income-types-alert'
          headingLevel='h3'
          type='info'
          condition={{ operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` }}
          batches={[`schedule-b-0`]}
        >
          <DFModal
            i18nKey='/info/income/interest/unsupported-interest-income-types-alert-modal'
            batches={[`schedule-b-0`]}
          />
        </DFAlert>
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='int-income-data-import-breather'>
        <Heading
          i18nKey='/heading/income/interest/data-import-breather'
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <Heading i18nKey='/heading/income/interest/data-import-breather-mfj' condition={`/isFilingStatusMFJ`} />
        <IntroContent i18nKey='/info/income/interest/data-import-breather-mfj' condition={`/isFilingStatusMFJ`} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='int-income-data-import'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/income/interest' />
        <Heading
          conditions={[
            { condition: `data-import`, section: `has-one-1099-int` },
            { operator: `isFalseOrIncomplete`, condition: `/isFilingStatusMFJ` },
          ]}
          i18nKey='/heading/income/interest/data-import-one-form'
        />
        <Heading
          conditions={[
            { condition: `data-import`, section: `has-one-1099-int`, operator: `isFalse` },
            { operator: `isFalseOrIncomplete`, condition: `/isFilingStatusMFJ` },
          ]}
          i18nKey='/heading/income/interest/data-import-multiple-forms'
        />
        <Heading
          conditions={[{ condition: `data-import`, section: `has-one-1099-int` }, { condition: `/isFilingStatusMFJ` }]}
          i18nKey='/heading/income/interest/data-import-one-form-mfj'
        />
        <Heading
          conditions={[
            { condition: `data-import`, section: `has-one-1099-int`, operator: `isFalse` },
            { condition: `/isFilingStatusMFJ` },
          ]}
          i18nKey='/heading/income/interest/data-import-multiple-forms-mfj'
        />
        <IntroContent i18nKey='/info/income/jobs-data-import-preview' />
        <DFModal
          i18nKey='/info/income/1099/what-if-1099-missing'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <DFModal condition={`/isFilingStatusMFJ`} i18nKey='/info/income/1099/what-if-1099-missing-mfj' />
        <DFModal i18nKey='/info/income/1099/how-does-irs-know' />
        <CollectionDataPreview
          subsubcategories={[`int-income-basic-info`, `int-income-amount`, `int-income-tax-withheld`]}
          nextRouteOverride={`/flow/income/interest/int-income-loop-intro`}
        />
      </Screen>
      <Screen route='int-income-data-import-breather-done'>
        <Heading
          i18nKey='/heading/income/interest/data-import-breather-done'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <Heading i18nKey='/heading/income/interest/data-import-breather-done-mfj' condition={`/isFilingStatusMFJ`} />
        <IntroContent
          i18nKey='/info/income/interest/data-import-breather-done'
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <IntroContent i18nKey='/info/income/interest/data-import-breather-done-mfj' condition={`/isFilingStatusMFJ`} />
        <DFModal
          i18nKey='/info/income/interest/why-do-have-to-add-spouse-1099-manually'
          condition={`/isFilingStatusMFJ`}
        />
        <SaveAndOrContinueButton nextRouteOverride='/data-view/flow/income/interest?reviewMode=true' />
      </Screen>
    </Gate>
    <Screen route='int-income-loop-intro'>
      <Heading i18nKey='/heading/income/interest' condition='/hasInterestReports' />
      <ContextHeading
        displayOnlyOn='edit'
        i18nKey='/heading/income/interest'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` }}
      />
      <Heading
        i18nKey='/heading/income/interest/intro'
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` },
        ]}
        batches={[`schedule-b-0`]}
      />
      <Heading
        i18nKey='/heading/income/interest/intro-mfj'
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` },
        ]}
        batches={[`schedule-b-0`]}
      />
      <IntroContent i18nKey='/info/income/interest/intro' />
      <ConditionalList
        i18nKey='/info/income/interest/intro-list'
        items={[
          {
            itemKey: `listItem_any_1099_forms`,
            conditions: [{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }],
          },
          { itemKey: `listItem_any_1099_forms-mfj`, conditions: [`/isFilingStatusMFJ`] },
          { itemKey: `listItem_information_taxable_interest` },
        ]}
      />
      <DFAlert
        i18nKey='/info/income/interest/unsupported-interest-income-types-alert'
        headingLevel='h3'
        type='info'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasInterestReports` }}
        batches={[`schedule-b-0`]}
      >
        <DFModal
          i18nKey='/info/income/interest/unsupported-interest-income-types-alert-modal'
          batches={[`schedule-b-0`]}
        />
      </DFAlert>
      <DFModal
        i18nKey='/info/income/interest/unsupported-interest-income-types'
        condition='/hasInterestReports'
        batches={[`schedule-b-0`]}
      />
      <DFModal i18nKey='/info/income/interest/taxable-interest-calculations' condition='/hasInterestReports' />
      <CollectionItemManager
        path='/interestReports'
        loopName='/interestReports'
        donePath='/interestReportsIsDone'
        batches={[`information-architecture-0`]}
      />
    </Screen>
    <CollectionLoop
      loopName='/interestReports'
      collection='/interestReports'
      iconName='AttachMoney'
      collectionItemCompletedCondition='/interestReports/*/isComplete'
      donePath='/interestReportsIsDone'
      isImportedFactPath='/interestReports/*/isImported'
      importedFlowStartRoute='/flow/income/interest/data-import-refer-int-income'
      importedFlowDonePath='/interestReports/*/hasSeenLastAvailableScreen'
      importedRouteOverride='/flow/income/interest/int-income-data-import-breather-done'
      dataViewSections={[
        {
          i18nKey: `dataviews./flow/income/interest.primaryFiler`,
          condition: `/interestReports/*/filer/isPrimaryFiler`,
        },
        {
          i18nKey: `dataviews./flow/income/interest.secondaryFiler`,
          condition: `/interestReports/*/filer/isSecondaryFiler`,
        },
      ]}
    >
      <Gate condition={`/interestReports/*/isImported`}>
        <Screen route='data-import-int-ko' condition={`/knockoutInterestIncome`} isKnockout={true}>
          <Heading i18nKey='/heading/income/interest-data-import-ko' />
          <InfoDisplay
            i18nKey='/info/income/interest-1099-has-info-we-do-not-support-singular'
            condition={{ condition: `data-import`, section: `has-one-1099-int` }}
          />
          <InfoDisplay
            i18nKey='/info/income/interest-1099-has-info-we-do-not-support-plural'
            condition={{ condition: `data-import`, section: `has-one-1099-int`, operator: `isFalse` }}
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/facta'
            condition='/interestReports/hasRequiredFactaFilings'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-2-ko'
            condition='/flowKnockoutHasEarlyWithdrawlPenalty'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-6-ko'
            condition='/flowKnockoutHasForeignTaxPaid'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-8-ko'
            condition='/flowKnockoutTaxExemptInterest'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-9-ko'
            condition='/interestReports/*/specifiedPrivateActivityBondInterest'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-10-ko'
            condition='/interestReports/*/marketDiscount'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-11-ko'
            condition='/interestReports/*/bondPremium'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-12-ko'
            condition='/interestReports/*/bondPremiumOnTreasuryObligations'
          />
          <DFAccordion
            i18nKey='/info/knockout/forms-missing/1099-int-box-13-ko'
            condition='/interestReports/*/bondPremiumOnTaxExemptBond'
          />
          <DFAlert
            className='margin-top-3'
            i18nKey='/info/knockout/data-import-file-using-another-tool'
            headingLevel='h2'
            type='warning'
          />
          <KnockoutButton i18nKey='button.knockout' />
          <SaveAndOrContinueButton
            isOutline={true}
            condition={{ condition: `data-import`, section: `has-one-1099-int` }}
            i18nKey='datapreviews./interestReports.button.reviewSingular'
            nextRouteOverride='/data-view/flow/income/interest'
          />
          <SaveAndOrContinueButton
            isOutline={true}
            condition={{ condition: `data-import`, section: `has-one-1099-int`, operator: `isFalse` }}
            i18nKey='datapreviews./interestReports.button.reviewPlural'
            nextRouteOverride='/data-view/flow/income/interest'
          />
        </Screen>
        <Screen route='data-import-breather-int-income'>
          <Heading i18nKey='/heading/income/interest-post-import-breather' />
          <InfoDisplay i18nKey='/info/income/jobs/interest-post-import-breather' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='data-import-refer-int-income' condition={`/interestReports/*/isImported`}>
          <Heading i18nKey='/heading/income/interest-data-import-refer-1099' />
          <InfoDisplay i18nKey='/info/income/interest-data-import-refer-1099' />
          <DFModal i18nKey='/info/income/interest-data-import-where-do-i-find-my-1099' />
          <SaveAndOrContinueButton nextRouteOverride='/flow/income/interest/1099-int-add-boxes-15-17' />
        </Screen>
      </Gate>
      <SubSubcategory route='int-income-basic-info'>
        <Screen route='add-int-income'>
          <Heading
            i18nKey='/heading/income/interest/add'
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading i18nKey='/heading/income/interest/add-mfj' condition='/isFilingStatusMFJ' />
          <DFModal i18nKey='/info/income/interest/what-if-nominee' batches={[`schedule-b-0`]} />
          <Boolean path='/interestReports/*/has1099' importedPath='/interestReports/*/importedHas1099' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Gate condition='/interestReports/*/has1099'>
        <SubSubcategory route='int-income-basic-info'>
          <Gate condition={{ operator: `isFalse`, condition: `/interestReports/*/isImported` }}>
            <Screen route='1099-int-add-whose' condition='/isFilingStatusMFJ'>
              <Heading i18nKey='/heading/income/interest/add-whose' />
              <TaxReturnAlert
                i18nKey='/info/income/interest/secondary-filer-income-without-mfj'
                headingLevel='h3'
                type='error'
                condition='/interestReports/*/secondaryFilerUsedWithoutMFJ'
              />
              <InfoDisplay i18nKey='/info/income/interest/add-whose' />
              <CollectionItemReference path='/interestReports/*/filer' displayOnlyOn='edit' />
              <GenericString path='/interestReports/*/filer/fullName' displayOnlyOn='data-view' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Screen route='1099-int-add-payer-name'>
            <Heading i18nKey='/heading/income/interest/add-payer-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-payer-1099' />
            <SetFactAction
              path='/interestReports/*/filer'
              source='/primaryFiler'
              conditions={[
                { operator: `isIncomplete`, condition: `/interestReports/*/filer` },
                { operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` },
              ]}
            />
            <SetFactAction
              path='/interestReports/*/filer'
              source='/primaryFiler'
              conditions={[
                // We can set the interest report's filer as the primary filer's if they are importing the 1099-INT.
                { operator: `isIncomplete`, condition: `/interestReports/*/filer` },
                { operator: `isTrue`, condition: `/interestReports/*/isImported` },
              ]}
            />
            <LimitingString path='/interestReports/*/payer' importedPath='/interestReports/*/importedPayer' />
            <LimitingString path='/interestReports/*/writablePayerNameLine2' required={false} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='add-int-facta'>
            <Heading i18nKey='/heading/income/interest/facta-1099' />
            <Boolean
              path='/interestReports/*/factaFilingRequired'
              importedPath='/interestReports/*/importedFactaFilingRequired'
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen
            route='1099-int-facta-knockout'
            condition='/interestReports/hasRequiredFactaFilings'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/forms-missing/facta' batches={[`schedule-b-0`]} />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='int-income-amount'>
          <Screen route='1099-int-add-box-1'>
            <Heading i18nKey='/heading/income/interest/add-box-1-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-box-1-1099' />
            <Dollar
              path='/interestReports/*/writable1099Amount'
              importedPath='/interestReports/*/importedBox1'
              required={false}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-int-add-box-2'>
            <Heading i18nKey='/heading/income/interest/add-box-2-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-box-2-1099' />
            <Dollar
              path='/interestReports/*/writableEarlyWithdrawlPenaltyAmount'
              importedPath='/interestReports/*/importedBox2'
              required={false}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-int-box-2-knockout' condition='/flowKnockoutHasEarlyWithdrawlPenalty' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/forms-missing/1099-int-box-2-ko' batches={[`schedule-b-0`]} />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
          <Screen route='1099-int-add-box-3'>
            <Heading i18nKey='/heading/income/interest/add-box-3-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-box-3-1099' />
            <DFModal i18nKey='/info/income/interest/exclude-under-ed-savings-bonds' batches={[`schedule-b-0`]} />
            <Dollar
              path='/interestReports/*/writableInterestOnGovernmentBonds'
              importedPath='/interestReports/*/importedBox3'
              required={false}
            />
            <Dollar path='/interestReports/*/taxableInterest' displayOnlyOn='data-view' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='int-income-tax-withheld'>
          <Screen route='1099-int-add-box-4'>
            <Heading i18nKey='/heading/income/interest/add-box-4-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-box-4-1099' />
            <Dollar
              path='/interestReports/*/writableTaxWithheld'
              importedPath='/interestReports/*/importedBox4'
              required={false}
            />
            <DFModal i18nKey='/info/income/interest/skip-box-5' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='payer-tin'>
          <Screen route='1099-int-add-payer-tin' condition='/interestReports/*/hasFederalWithholding'>
            <Heading i18nKey='/heading/income/interest/1099-add-payer-tin' />
            <Ein path='/interestReports/*/payer/tin' importedPath='/interestReports/*/importedPayerTin' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='foreign-tax-paid'>
          <Screen route='1099-int-add-box-6'>
            <Heading i18nKey='/heading/income/interest/add-box-6-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-box-6-1099' />
            <Dollar
              path='/interestReports/*/writableForeignTaxPaid'
              importedPath='/interestReports/*/importedBox6'
              required={false}
              batches={[`schedule-b-0`]}
            />
            <DFModal i18nKey='/info/income/interest/skip-box-7' batches={[`schedule-b-0`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-int-box-6-knockout' condition='/flowKnockoutHasForeignTaxPaid' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/forms-missing/1099-int-box-6-ko' batches={[`schedule-b-0`]} />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='int-income-tax-exempt'>
          <Screen route='1099-int-add-box-8'>
            <Heading i18nKey='/heading/income/interest/add-box-8-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-box-8-1099' />
            <Dollar
              path='/interestReports/*/writableTaxExemptInterest'
              importedPath='/interestReports/*/importedBox8'
              required={false}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-int-box-8-ko' condition='/flowKnockoutTaxExemptInterest' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              batches={[`income-assorted-kos-0`, `schedule-b-0`]}
              i18nKey='/heading/knockout/forms-missing/1099-int-box-8-ko'
            />
            <DFAlert
              batches={[`income-assorted-kos-0`, `schedule-b-0`]}
              i18nKey='/info/knockout/generic-other-ways-to-file'
              headingLevel='h2'
              type='warning'
            />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='int-income-bonds'>
          <Screen route='1099-int-add-boxes-9-14'>
            <Heading i18nKey='/heading/income/interest/add-boxes-9-14-1099' batches={[`schedule-b-0`]} />
            <InfoDisplay i18nKey='/info/income/interest/add-boxes-9-14-1099' />
            <Dollar
              path='/interestReports/*/writableSpecifiedPrivateActivityBondInterest'
              importedPath='/interestReports/*/importedBox9'
              required={false}
            />
            <Dollar
              path='/interestReports/*/writableMarketDiscount'
              importedPath='/interestReports/*/importedBox10'
              required={false}
            />
            <Dollar
              path='/interestReports/*/writableBondPremium'
              importedPath='/interestReports/*/importedBox11'
              required={false}
            />
            <Dollar
              path='/interestReports/*/writableBondPremiumOnTreasuryObligations'
              importedPath='/interestReports/*/importedBox12'
              required={false}
            />
            <Dollar
              path='/interestReports/*/writableBondPremiumOnTaxExemptBond'
              importedPath='/interestReports/*/importedBox13'
              required={false}
            />
            <GenericString
              path='/interestReports/*/writableTaxExemptAndTaxCreditBondCusipNo'
              importedPath='/interestReports/*/importedBox14'
              required={false}
            />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-int-boxes-9-13-knockout' condition='/flowKnockoutHasBonds' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/forms-missing/1099-int-box-9-13-ko' batches={[`schedule-b-0`]} />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='int-income-state-info'>
          <Screen route='1099-int-add-boxes-15-17'>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/income/interest-data-import-context'
              condition='/interestReports/*/isImported'
            />
            <Heading i18nKey='/heading/income/interest/add-boxes-15-17-1099' />
            <InfoDisplay i18nKey='/info/income/interest/add-boxes-15-17-1099' />
            <Enum path='/interestReports/*/writableState' renderAs='select' required={false} />
            <GenericString path='/interestReports/*/writableStateIdNumber' required={false} />
            <Dollar path='/interestReports/*/writableStateTaxWithheld' required={false} />
            <SetFactAction path='/interestReports/*/hasSeenLastAvailableScreen' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='1099-int-box-17-ko' condition='/flowKnockoutStateTaxWithheld' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading
              i18nKey='/heading/knockout/forms-missing/1099-int-state-tax-withheld-ko'
              batches={[`schedule-b-0`]}
            />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
          <Screen
            route='1099-int-other-state-knockout'
            condition='/flowKnockoutWithholdingUnsupportedState'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/forms-missing/1099-int-other-state-ko' />
            <InfoDisplay i18nKey='/info/knockout/income/interest/1099-int-other-state-ko' />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </SubSubcategory>
      </Gate>
      <Gate condition={{ operator: `isFalse`, condition: `/interestReports/*/has1099` }}>
        <SubSubcategory route='int-income-basic-info'>
          <Screen route='no-1099-int-add-whose' condition='/isFilingStatusMFJ'>
            <Heading i18nKey='/heading/income/interest/add-whose' />
            <InfoDisplay i18nKey='/info/income/interest/no-add-whose' />
            <CollectionItemReference path='/interestReports/*/filer' displayOnlyOn='edit' />
            <GenericString path='/interestReports/*/filer/fullName' displayOnlyOn='data-view' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='no-1099-int-add-payer-name'>
            <Heading i18nKey='/heading/income/interest/add-payer' />
            <InfoDisplay i18nKey='/info/income/interest/no-add-payer' />
            <SetFactAction
              path='/interestReports/*/filer'
              source='/primaryFiler'
              conditions={[
                { operator: `isIncomplete`, condition: `/interestReports/*/filer` },
                { operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` },
              ]}
            />
            <LimitingString path='/interestReports/*/payer' />
            <LimitingString path='/interestReports/*/writablePayerNameLine2' required={false} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='int-income-amount'>
          <Screen route='no-1099-int-taxable-int'>
            <Heading i18nKey='/heading/income/interest/taxable-int' batches={[`schedule-b-0`]} />
            <DFModal i18nKey='/info/income/interest/where-to-find-taxable-interest' />
            <Dollar path='/interestReports/*/no1099Amount' />
            <SetFactAction path='/interestReports/*/hasSeenLastAvailableScreen' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Gate>
    </CollectionLoop>
  </Subcategory>
);
