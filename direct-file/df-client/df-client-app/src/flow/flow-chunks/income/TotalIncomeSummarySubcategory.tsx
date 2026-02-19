import {
  Heading,
  SummaryTable,
  InfoDisplay,
  SetFactAction,
  SaveAndOrContinueButton,
  DFAlert,
} from '../../ContentDeclarations.js';
import { Screen, Subcategory } from '../../flowDeclarations.js';

export const TotalIncomeSummarySubcategory = (
  <Subcategory route='total-income-summary' completeIf='/flowHasSeenTotalIncomeSummary' skipDataView>
    <Screen route='total-income-summary' condition={`/flowHasReportedIncome`}>
      <Heading i18nKey='/heading/income/total-income-summary' batches={[`information-architecture-0`]} />
      <SummaryTable
        i18nKey='/info/income/total-income-summary'
        items={[
          { itemKey: `jobs`, conditions: [`/hasW2s`] },
          { itemKey: `jobsAmount`, conditions: [`/hasW2s`] },
          { itemKey: `unemployment`, conditions: [`/receivedUnemploymentCompensation`] },
          { itemKey: `unemploymentAmount`, conditions: [`/receivedUnemploymentCompensation`] },
          { itemKey: `interest`, conditions: [`/hasInterestReports`] },
          { itemKey: `interestAmount`, conditions: [`/hasInterestReports`] },
          { itemKey: `alaskaPfd`, conditions: [`/has1099Misc`] },
          { itemKey: `alaskaPfdAmount`, conditions: [`/has1099Misc`] },
          { itemKey: `dependentCare`, conditions: [`/hasReportedDependentCareBenefits`] },
          { itemKey: `dependentCareAmount`, conditions: [`/hasReportedDependentCareBenefits`] },
          { itemKey: `retirement`, conditions: [`/has1099R`] },
          { itemKey: `retirementAmount`, conditions: [`/has1099R`] },
          { itemKey: `socialSecurity`, conditions: [`/hasSocialSecurityBenefits`] },
          { itemKey: `socialSecurityAmount`, conditions: [`/hasSocialSecurityBenefits`] },
          { itemKey: `total`, showTopBorder: true },
        ]}
        batches={[`information-architecture-0`]}
      />
      <InfoDisplay i18nKey='/info/income/total-income-summary-explainer' batches={[`information-architecture-0`]} />

      <SetFactAction
        source='/flowTrue'
        path='/flowHasSeenTotalIncomeSummary'
        batches={[`information-architecture-0`]}
      />

      <SaveAndOrContinueButton />
    </Screen>

    <Screen
      route='total-income-summary-none-reported'
      condition={{ operator: `isFalseOrIncomplete`, condition: `/flowHasReportedIncome` }}
    >
      <Heading i18nKey='/heading/income/total-income-summary-none-reported' batches={[`information-architecture-0`]} />

      <InfoDisplay i18nKey='/info/income/total-income-summary-none-reported' batches={[`information-architecture-0`]} />

      <DFAlert
        headingLevel='h3'
        type='info'
        i18nKey='/info/income/total-income-summary-none-reported-alert'
        batches={[`information-architecture-0`]}
      />

      <SetFactAction
        source='/flowTrue'
        path='/flowHasSeenTotalIncomeSummary'
        batches={[`information-architecture-0`]}
      />

      <SaveAndOrContinueButton />
    </Screen>
  </Subcategory>
);
