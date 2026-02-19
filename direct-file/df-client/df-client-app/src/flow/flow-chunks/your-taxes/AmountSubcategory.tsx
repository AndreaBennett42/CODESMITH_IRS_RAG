/* eslint-disable max-len */
import { Screen, Subcategory } from '../../flowDeclarations.js';
import {
  BigContent,
  ContextHeading,
  DFModal,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
  SetFactAction,
  SummaryTable,
  TaxReturnAlert,
} from '../../ContentDeclarations.js';

export const AmountSubcategory = (
  <Subcategory route='amount' completeIf='/flowHasSeenAmount'>
    <Screen route='tax-amount-intro'>
      <ContextHeading displayOnlyOn='edit' i18nKey='/heading/your-taxes/amount/tax-amount' />
      <Heading i18nKey='/heading/your-taxes/amount/tax-amount-intro' />
      <TaxReturnAlert
        type='warning'
        i18nKey='/info/your-taxes/amount/tax-amount-has-changed'
        conditions={[{ operator: `isTrueAndComplete`, condition: `/flowHasAmountChanged` }]}
        checklistSubcategoryWarningLabel='/info/your-taxes/amount/tax-amount-has-changed'
      />
      <InfoDisplay i18nKey='/info/your-taxes/amount/tax-amount-intro' />
      <SaveAndOrContinueButton />
    </Screen>
    <Screen route='tax-amount-number'>
      <Heading
        i18nKey='/heading/your-taxes/amount/tax-amount-number-owed'
        condition={{ operator: `isFalse`, condition: `/dueRefund` }}
      />
      <Heading i18nKey='/heading/your-taxes/amount/tax-amount-number-refund' condition='/dueRefund' />
      <BigContent
        i18nKey='/info/your-taxes/amount/tax-amount-number-owed'
        condition={{ operator: `isFalse`, condition: `/dueRefund` }}
      />
      <BigContent i18nKey='/info/your-taxes/amount/tax-amount-number-refund' condition='/dueRefund' />
      <DFModal i18nKey='/info/your-taxes/amount/irs-may-adjust' />
      <DFModal i18nKey='/info/your-taxes/amount/tax-amount-means-to-owe' condition='/owesBalance' />
      <DFModal i18nKey='/info/your-taxes/amount/tax-amount-means-to-refund' condition='/dueRefund' />
      <DFModal i18nKey='/info/your-taxes/amount/tax-amount-means-to-zero' condition='/zeroBalance' />
      <DFModal
        i18nKey='/info/your-taxes/amount/tax-amount-withholdings-effect'
        condition={{ operator: `isFalse`, condition: `/zeroBalance` }}
      />
      <SaveAndOrContinueButton />
    </Screen>
    <Screen route='tax-amount-explanation' actAsDataView={true}>
      <Heading i18nKey='/heading/your-taxes/amount/tax-amount-explanation' />
      <SummaryTable
        i18nKey='/info/your-taxes/amount/tax-amount-explanation-total-income'
        items={[
          {
            itemKey: `totalIncome`,
          },
        ]}
      />
      <SummaryTable
        i18nKey='/info/your-taxes/amount/tax-amount-explanation-adjustments'
        items={[
          {
            itemKey: `adjustmentsSubheader`,
            conditions: [`/hasAdjustmentsToIncome`],
          },
          {
            itemKey: `hsaDeductions`,
            conditions: [`/hasHsaDeduction`],
            indent: true,
          },
          {
            itemKey: `educatorAdjustments`,
            conditions: [`/isReceivingEducatorExpensesAdjustment`],
            indent: true,
          },
          {
            itemKey: `studentLoanDeduction`,
            conditions: [`/isReceivingStudentLoanInterestAdjustment`],
            indent: true,
          },
          {
            itemKey: `agi`,
            showTopBorder: true,
          },
          {
            itemKey: `standardDeduction`,
          },
          {
            itemKey: `taxableIncome`,
            showTopBorder: true,
          },
        ]}
      />
      <SummaryTable
        i18nKey='/info/your-taxes/amount/tax-amount-explanation-tax-amount'
        items={[
          {
            itemKey: `initialTaxAmount`,
            showTopBorder: true,
          },
        ]}
      />
      <SummaryTable
        i18nKey='/info/your-taxes/amount/tax-amount-explanation-with-additional-taxes'
        items={[
          {
            itemKey: `excessAdvancePTC`,
          },
          {
            itemKey: `addtionalTaxes`,
            showTopBorder: true,
          },
        ]}
        condition={`/owesPtc`}
      />
      <SummaryTable
        i18nKey='/info/your-taxes/amount/tax-amount-explanation-credits'
        batches={[`ptc-1`]}
        items={[
          {
            itemKey: `nonrefundableCredits`,
            conditions: [`/isReceivingNonrefundableCredits`],
          },
          {
            itemKey: `cdcc`,
            conditions: [`/isReceivingCdccCredit`],
            indent: true,
          },
          {
            itemKey: `edc`,
            indent: true,
            conditions: [`/isReceivingEdc`],
          },
          {
            itemKey: `odc`,
            conditions: [`/isReceivingOdc`],
            indent: true,
          },
          {
            itemKey: `ctc`,
            conditions: [`/isReceivingCtc`],
            indent: true,
          },
          {
            itemKey: `savers`,
            conditions: [`/isReceivingSaversCredit`],
            indent: true,
          },
          {
            itemKey: `refundableCredits`,
            conditions: [`/isReceivingRefundableCredits`],
          },
          {
            itemKey: `ptcPositive`,
            conditions: [`/isReceivingPtc`],
            indent: true,
          },
          {
            itemKey: `actc`,
            conditions: [`/isReceivingActc`],
            indent: true,
          },
          {
            itemKey: `eitc`,
            conditions: [`/isReceivingEitc`],
            indent: true,
          },
          {
            itemKey: `taxAmount`,
            showTopBorder: true,
          },
        ]}
      />
      <SummaryTable
        i18nKey='/info/your-taxes/amount/tax-amount-explanation-final'
        items={[
          {
            itemKey: `withholdings`,
          },
          {
            itemKey: `estimated`,
          },
          {
            itemKey: `taxAmount`,
            showTopBorder: true,
          },
        ]}
      />
      <InfoDisplay i18nKey='/info/your-taxes/amount/tax-amount-explanation-refund' condition='/dueRefund' />
      <InfoDisplay i18nKey='/info/your-taxes/amount/tax-amount-explanation-owe' condition='/owesBalance' />
      <InfoDisplay i18nKey='/info/your-taxes/amount/tax-amount-explanation-zero' condition='/zeroBalance' />
      <SetFactAction path='/flowHasSeenAmount' source='/flowTrue' />
      <SetFactAction path='/flowHasAmountChanged' source='/flowFalse' />
      <SaveAndOrContinueButton />
    </Screen>
  </Subcategory>
);
