import { CURRENT_TAX_YEAR } from '../../../constants/taxConstants.js';
import { render, screen } from '@testing-library/react';
import ReturnErrorScreen, { ReturnErrorScreenProps } from './ReturnErrorScreen.js';
import { wrapComponent } from '../../../test/helpers.js';

const defaultProps: ReturnErrorScreenProps = {
  taxYear: parseInt(CURRENT_TAX_YEAR),
};
describe(`ReturnErrorScreen`, () => {
  const renderReturnErrorScreen = (props: ReturnErrorScreenProps) => {
    render(wrapComponent(<ReturnErrorScreen {...props} />));

    const heading = screen.getByRole(`heading`, { name: `heading`, level: 1 });
    const content = screen.getByText(`content`);

    return { heading, content };
  };

  it(`Renders without error`, () => {
    const { heading, content } = renderReturnErrorScreen(defaultProps);

    expect(heading).toBeInTheDocument();
    expect(content).toBeInTheDocument();
  });
});
