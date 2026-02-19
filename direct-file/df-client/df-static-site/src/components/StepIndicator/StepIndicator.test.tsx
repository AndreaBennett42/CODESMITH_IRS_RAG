import { render, screen } from '@testing-library/react';

import StepIndicator from './StepIndicator.js';

describe(`StepIndicator component`, () => {
  test(`renders without errors`, () => {
    render(<StepIndicator currentStepKey='ScreenerState' />);

    expect(screen.getByRole(`list`)).toBeInTheDocument();
    expect(screen.getAllByRole(`listitem`)[0]).toHaveClass(`usa-step-indicator__segment--current`);
    expect(screen.getByText(`1`)).toHaveClass(`usa-step-indicator__current-step`);
  });

  test(`renders second step current and first step complete`, () => {
    render(<StepIndicator currentStepKey='ScreenerIncome' />);

    expect(screen.getByRole(`list`)).toBeInTheDocument();
    expect(screen.getAllByRole(`listitem`)[0]).toHaveClass(`usa-step-indicator__segment--complete`);
    expect(screen.getAllByRole(`listitem`)[1]).toHaveClass(`usa-step-indicator__segment--current`);
    expect(screen.getByText(`2`)).toHaveClass(`usa-step-indicator__current-step`);
  });

  test(`renders third step current and fourth step incomplete`, () => {
    render(<StepIndicator currentStepKey='ScreenerSavingsAndRetirement' />);

    expect(screen.getByRole(`list`)).toBeInTheDocument();
    expect(screen.getAllByRole(`listitem`)[2]).toHaveClass(`usa-step-indicator__segment--current`);
    expect(screen.getAllByRole(`listitem`)[3]).toHaveClass(`usa-step-indicator__segment`);
    expect(screen.getByText(`3`)).toHaveClass(`usa-step-indicator__current-step`);
  });
});
