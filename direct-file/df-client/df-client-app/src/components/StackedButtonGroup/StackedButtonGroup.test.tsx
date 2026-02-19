import { render, screen, within } from '@testing-library/react';
import { Button } from '@trussworks/react-uswds';

import StackedButtonGroup from './StackedButtonGroup.js';
describe(`StackedButtonGroup component`, () => {
  it(`Renders the buttons stacked within the parent element`, () => {
    render(
      <StackedButtonGroup>
        <Button type='submit'>Button 1</Button>
        <Button type='button'>Button 2</Button>
      </StackedButtonGroup>
    );

    const parentDiv = screen.getByTestId(`stackedButtonGroup`);
    const button1 = within(parentDiv).queryByRole(`button`, { name: `Button 1` });
    const button2 = within(parentDiv).queryByRole(`button`, { name: `Button 2` });

    expect(button1).toBeInTheDocument();
    expect(button2).toBeInTheDocument();
  });
});
