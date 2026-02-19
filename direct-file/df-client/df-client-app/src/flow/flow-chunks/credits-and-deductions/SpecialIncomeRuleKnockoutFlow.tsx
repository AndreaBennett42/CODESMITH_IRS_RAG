import { AbsolutePath } from '../../../fact-dictionary/Path.js';
import { BATCH_NAME } from '../../batches.js';
import { Gate, Screen } from '../../flowDeclarations.js';
import {
  Boolean,
  ContextHeading,
  DFAlert,
  DFModal,
  Heading,
  IconDisplay,
  InfoDisplay,
  KnockoutButton,
  SaveAndOrContinueButton,
} from '../../ContentDeclarations.js';

type cdccSubSectionNames = {
  CREDITS: string;
  DEPENDENT_CARE_BENEFITS: string;
};

export const cdccSubSectionNames: cdccSubSectionNames = {
  DEPENDENT_CARE_BENEFITS: `benefits`,
  CREDITS: `credits`,
};

export const earnedIncomeRuleKnockoutCheckFlow = (
  suffix: string,
  gateCondition: AbsolutePath,
  batches: BATCH_NAME[]
) => {
  let breatherScreen: JSX.Element = (
    <Screen route='earned-income-rule-breather-benefits'>
      <ContextHeading
        i18nKey='/heading/credits-and-deductions/credits/cdcc'
        condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'
        batches={[`cdcc-0`]}
      />
      <Heading i18nKey='/heading/income/dependent-care/earned-income-rule-breather-benefits' batches={batches} />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/earned-income-rule-breather-benefits-student-only'
        conditions={[`/isStudentOnly`, `/isFilingStatusMFJ`]}
        batches={batches}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/earned-income-rule-breather-benefits-disabled-only'
        conditions={[`/isDisabledOnly`, `/isFilingStatusMFJ`]}
        batches={batches}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/earned-income-rule-breather-benefits-student-and-disabled'
        conditions={[`/isStudentAndDisabled`, `/isFilingStatusMFJ`]}
        batches={batches}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/earned-income-rule-breather-benefits-student-only-MFS'
        conditions={[`/isStudentOnly`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        batches={batches}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/earned-income-rule-breather-benefits-disabled-only-MFS'
        conditions={[`/isDisabledOnly`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        batches={batches}
      />
      <InfoDisplay
        i18nKey='/info/income/dependent-care/earned-income-rule-breather-benefits-student-and-disabled-MFS'
        conditions={[`/isStudentAndDisabled`, { operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        batches={batches}
      />
      <DFModal
        i18nKey={`/info/income/dependent-care/earned-income-rule-breather/which-situation-changes-${suffix}`}
        batches={[`cdcc-2`]}
      />
      <SaveAndOrContinueButton />
    </Screen>
  );
  let earnedIncomeRuleResultScreen: JSX.Element = (
    <Screen route={`earned-income-rule-result-${suffix}`} condition='/neitherTpEarnedLessThanSpecialEarnedIncomeMax'>
      <ContextHeading
        i18nKey='/heading/credits-and-deductions/credits/cdcc'
        condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'
        batches={[`cdcc-0`]}
      />
      <Heading
        i18nKey={`/heading/income/dependent-care/earned-income-rule-result-${suffix}`}
        condition='/isFilingStatusMFJ'
        batches={batches}
      />
      <Heading
        i18nKey={`/heading/income/dependent-care/earned-income-rule-result-${suffix}-MFS`}
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        batches={batches}
      />
      <DFModal
        i18nKey={`/info/income/dependent-care/earned-income-rule-breather/which-situation-changes-${suffix}`}
        batches={[`cdcc-2`]}
      />
      <SaveAndOrContinueButton />
    </Screen>
  );

  if (suffix === cdccSubSectionNames.CREDITS) {
    breatherScreen = (
      <Screen route={`earned-income-rule-breather-credits`}>
        <ContextHeading
          displayOnlyOn='edit'
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'
          batches={[`cdcc-0`]}
        />
        <Heading i18nKey='/heading/income/dependent-care/earned-income-rule-breather-credits' batches={batches} />
        <InfoDisplay
          i18nKey={`/info/income/dependent-care/earned-income-rule-breather-credits-student-only`}
          condition='/isStudentOnly'
          batches={batches}
        />
        <InfoDisplay
          i18nKey={`/info/income/dependent-care/earned-income-rule-breather-credits-disabled-only`}
          condition='/isDisabledOnly'
          batches={batches}
        />
        <InfoDisplay
          i18nKey={`/info/income/dependent-care/earned-income-rule-breather-credits-student-and-disabled`}
          condition='/isStudentAndDisabled'
          batches={batches}
        />
        <DFModal
          i18nKey={`/info/income/dependent-care/earned-income-rule-breather/which-situation-changes-credits`}
          batches={[`cdcc-2`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    );
    earnedIncomeRuleResultScreen = (
      <Screen route={`earned-income-rule-result-${suffix}`} condition='/neitherTpEarnedLessThanSpecialEarnedIncomeMax'>
        <ContextHeading
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'
          batches={[`cdcc-0`]}
        />
        <Heading i18nKey={`/heading/income/dependent-care/earned-income-rule-result-${suffix}`} batches={batches} />
        <DFModal
          i18nKey={`/info/income/dependent-care/earned-income-rule-breather/which-situation-changes-${suffix}`}
          batches={[`cdcc-2`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
    );
  }

  return (
    <Gate condition={gateCondition}>
      {breatherScreen}
      <Screen route={`earned-income-rule-tp1-${suffix}`} condition='/primaryFiler/isStudentOrDisabled'>
        <ContextHeading
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-tp1-student-only'
          condition='/primaryFiler/isStudentOnly'
          batches={batches}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-tp1-disabled-only'
          condition='/primaryFiler/isDisabledOnly'
          batches={batches}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-tp1-student-and-disabled'
          condition='/primaryFiler/isStudentAndDisabled'
          batches={batches}
        />
        <DFModal
          i18nKey={`/info/income/dependent-care/earned-income-rule-tp1/why-ask-if-made-under-${suffix}`}
          batches={batches}
        />
        <Boolean path='/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome' batches={[`cdcc-2`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route={`earned-income-rule-tp2-${suffix}`} condition='/showEarnedIncomeRuleTp2'>
        <ContextHeading
          i18nKey='/heading/credits-and-deductions/credits/cdcc'
          condition='/showSpecialEarnedIncomeKnockoutFlowInCredits'
          batches={[`cdcc-0`]}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-tp2-student-only'
          condition='/secondaryFiler/isStudentOnly'
          batches={batches}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-tp2-disabled-only'
          condition='/secondaryFiler/isDisabledOnly'
          batches={batches}
        />
        <Heading
          i18nKey='/heading/income/dependent-care/earned-income-rule-tp2-student-and-disabled'
          condition='/secondaryFiler/isStudentAndDisabled'
          batches={batches}
        />
        <DFModal
          i18nKey={`/info/income/dependent-care/earned-income-rule-tp1/why-ask-if-made-under-${suffix}`}
          batches={batches}
        />
        <Boolean path='/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome' batches={[`cdcc-2`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route={`earned-income-rule-ko-${suffix}`} condition='/knockoutStudentOrDisabled' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/income/dependent-care/earned-income-rule-knockout' batches={batches} />
        <InfoDisplay i18nKey={`/info/income/dependent-care/earned-income-rule-knockout-${suffix}`} batches={batches} />
        <DFModal
          i18nKey={`/info/income/dependent-care/earned-income-rule-breather/which-situation-changes-${suffix}`}
          batches={[`cdcc-2`]}
        />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
      {earnedIncomeRuleResultScreen}
    </Gate>
  );
};
