import { StepIndicator as USAStepIndicator, StepIndicatorStep } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';

const stepKeys = [
  `ScreenerState`,
  `ScreenerIncome`,
  `ScreenerSavingsAndRetirement`,
  `ScreenerDeductions`,
  `ScreenerCredits`,
] as const;
type currentStepKeyType = (typeof stepKeys)[number];

type StepIndicatorProps = {
  currentStepKey: currentStepKeyType;
};

type StepIndicatorStepStatus = 'complete' | 'current' | 'incomplete';

const getStepStatus = (index: number, currentStepIndex: number): StepIndicatorStepStatus => {
  if (index === currentStepIndex) return `current`;
  if (index < currentStepIndex) return `complete`;
  return `incomplete`;
};

const StepIndicator = ({ currentStepKey }: StepIndicatorProps) => {
  const { t } = useTranslation(`translation`);

  const indexOfCurrentStep = stepKeys.findIndex((label) => label === currentStepKey);

  return (
    <USAStepIndicator
      showLabels={false}
      headingLevel='h1'
      ofText={t(`components.stepIndicator.ofText`)}
      stepText={t(`components.stepIndicator.stepText`)}
    >
      {stepKeys.map((page, index) => {
        return (
          <StepIndicatorStep
            key={index}
            status={getStepStatus(index, indexOfCurrentStep)}
            label={t(`pages.${page}.stepIndicator.label`)}
          />
        );
      })}
    </USAStepIndicator>
  );
};

export default StepIndicator;
