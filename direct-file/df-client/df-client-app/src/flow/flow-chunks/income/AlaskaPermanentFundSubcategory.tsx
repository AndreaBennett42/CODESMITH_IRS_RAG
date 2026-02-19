/* eslint-disable max-len */
import { CollectionLoop, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
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
} from '../../ContentDeclarations.js';

export const AlaskaPermanentFundSubcategory = (
  /* Notes
  - Need to verify completeIf and collectionContext
  - Let's work with apf route for now
  */
  <Subcategory
    route='alaska-pfd'
    completeIf='/isApfSectionComplete'
    collectionContext='/form1099Miscs'
    displayOnlyIf='/eligibleForApf'
    dataItems={[
      {
        itemKey: `alaskaPfdTaxable`,
        conditions: [`/hasAlaskaPfdIncome`],
      },
      {
        itemKey: `alaskaPfdNoneTaxable`,
        conditions: [{ operator: `isFalse`, condition: `/hasAlaskaPfdIncome` }],
      },
    ]}
  >
    <Screen route='pfd-loop-intro'>
      <ContextHeading
        displayOnlyOn='edit'
        i18nKey='/heading/income/alaska-pfd'
        batches={[`alaska-permanent-fund-0`]}
        condition={{ operator: `isFalseOrIncomplete`, condition: `/has1099Misc` }}
      />
      <Heading i18nKey='/heading/income/alaska-pfd' batches={[`alaska-permanent-fund-0`]} condition='/has1099Misc' />

      <Heading
        i18nKey='/heading/income/alaska-pfd/intro'
        batches={[`alaska-permanent-fund-0`]}
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/has1099Misc` },
        ]}
      />
      <Heading
        i18nKey='/heading/income/alaska-pfd/intro-mfj'
        batches={[`alaska-permanent-fund-0`]}
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/has1099Misc` },
        ]}
      />

      <InfoDisplay
        condition='/hasAtLeastOneCompletedApfIncomeItem'
        i18nKey='/info/income/pfd/add'
        batches={[`alaska-permanent-fund-0`]}
      />

      <DFModal
        condition='/hasNoCompleteApfIncomeItems'
        i18nKey='/info/income/pfd/report-child-pfd-income'
        batches={[`alaska-permanent-fund-0`]}
      />

      <DFModal
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        i18nKey='/info/income/pfd/report-if-garnished-modal'
        batches={[`alaska-permanent-fund-0`]}
      />

      <DFModal
        condition='/isFilingStatusMFJ'
        i18nKey='/info/income/pfd/report-if-garnished-modal-mfj'
        batches={[`alaska-permanent-fund-0`]}
      />

      <DFAlert
        condition='/hasNoCompleteApfIncomeItems'
        i18nKey='/info/income/pfd/supported-income-alert'
        headingLevel='h2'
        batches={[`alaska-permanent-fund-0`]}
        type='info'
      >
        <DFModal i18nKey='/info/income/pfd/supported-income-alert-modal' batches={[`alaska-permanent-fund-0`]} />
      </DFAlert>

      <DFModal
        condition='/hasAtLeastOneCompletedApfIncomeItem'
        i18nKey='/info/income/pfd/supported-income-alert-modal'
        batches={[`alaska-permanent-fund-0`]}
      />

      <CollectionItemManager
        path='/form1099Miscs'
        loopName='/form1099Miscs'
        donePath='/hasCompletedApfSection'
        batches={[`information-architecture-0`]}
      />
    </Screen>
    <CollectionLoop
      loopName='/form1099Miscs'
      collection='/form1099Miscs'
      collectionItemCompletedCondition='/form1099Miscs/*/isComplete'
      donePath='/hasCompletedApfSection'
      dataViewSections={[
        {
          i18nKey: `dataviews./flow/income/1099Misc.has1099Misc`,
          condition: `/form1099Miscs/*/filer/isPrimaryFiler`,
        },
        {
          i18nKey: `dataviews./flow/income/1099Misc.secondaryFiler1099Misc`,
          condition: `/form1099Miscs/*/belongsToSecondaryFiler`,
        },
      ]}
    >
      <SubSubcategory route='pfd-basic-info'>
        <Screen route='pfd-add-whose-1099' condition='/isFilingStatusMFJ'>
          <TaxReturnAlert
            i18nKey='/info/income/pfd/secondary-filer-income-without-mfj'
            headingLevel='h3'
            type='error'
            condition='/form1099Miscs/*/secondaryFilerUsedWithoutMFJ'
            batches={[`alaska-permanent-fund-0`]}
          />
          <Heading i18nKey='/heading/income/1099Misc/1099Misc-add-whose-pfd' batches={[`alaska-permanent-fund-0`]} />
          <DFModal i18nKey='/info/income/pfd/report-child-pfd-income' batches={[`alaska-permanent-fund-0`]} />
          <CollectionItemReference
            path='/form1099Miscs/*/filer'
            displayOnlyOn='edit'
            batches={[`alaska-permanent-fund-0`]}
          />
          <GenericString
            path='/form1099Miscs/*/filer/firstName'
            displayOnlyOn='data-view'
            batches={[`alaska-permanent-fund-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='pfd-add-box-3'>
          <Heading i18nKey='/heading/income/1099Misc/1099Misc-add-box-3' batches={[`alaska-permanent-fund-0`]} />
          <InfoDisplay i18nKey='/info/income/1099Misc/box-3-amounts-summary' />
          <Dollar path='/form1099Miscs/*/writableOtherIncome' batches={[`alaska-permanent-fund-0`]} />
          {/* Need to set the value here incase the placeholder wasn't edited */}
          <SetFactAction path='/alaskaPfd1099s/*/writableOtherIncome' source='/form1099Miscs/*/writableOtherIncome' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='pfd-add-box-4'>
          <Heading i18nKey='/heading/income/1099Misc/1099Misc-add-box-4' batches={[`alaska-permanent-fund-0`]} />
          <InfoDisplay i18nKey='/info/income/1099Misc/box-4-can-be-blank' batches={[`alaska-permanent-fund-0`]} />
          <Dollar
            path='/form1099Miscs/*/writableFederalWithholding'
            required={false}
            batches={[`alaska-permanent-fund-0`]}
          />
          <InfoDisplay
            i18nKey='/info/income/1099Misc/why-are-we-skipping-other-info'
            batches={[`alaska-permanent-fund-0`]}
          />
          <DFModal
            i18nKey='/info/income/pfd/why-are-we-skipping-other-info-modal'
            batches={[`alaska-permanent-fund-0`]}
          />
          <SetFactAction path='/form1099Miscs/*/payer' source='/apfForm1099MiscPayer' />
          <SetFactAction path='/form1099Miscs/*/payer/tin' source='/apfForm1099MiscPayerEin' />
          <SetFactAction path='/form1099Miscs/*/hasSeenLastAvailableScreen' source='/flowTrue' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
    </CollectionLoop>
  </Subcategory>
);
