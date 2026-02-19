import PaperPathStatusAlert from './PaperPathStatusAlert.js';
import { render, screen } from '@testing-library/react';

describe(PaperPathStatusAlert.name, () => {
  const renderPaperPathStatusAlert = () => {
    render(<PaperPathStatusAlert />);

    const alert = screen.getByTestId(`paper-path-status-alert`);

    return {
      alert,
    };
  };

  it(`renders correctly`, () => {
    const { alert } = renderPaperPathStatusAlert();

    expect(alert).toBeInTheDocument();
    expect(alert).toHaveClass(`usa-alert--warning`);
    expect(alert).toHaveTextContent(`paperPathStatusAlert.heading`);
    expect(alert).toHaveTextContent(`paperPathStatusAlert.body`);
  });
});
