import { render, screen } from '@testing-library/react';
import IconDisplay from './IconDisplay.js';

describe(`IconDisplay`, () => {
  test(`renders icon with all given props`, () => {
    render(
      <IconDisplay
        name='Construction'
        size={6}
        className='custom-style'
        style={{ color: `red` }}
        i18nKey='info/path/to/key'
      />
    );
    const iconElement = screen.getByLabelText(`info/path/to/key`);
    expect(iconElement).toBeInTheDocument();
    expect(iconElement).toHaveClass(`usa-icon--size-6`);
    expect(iconElement).toHaveClass(`custom-style`);
    expect(iconElement).toHaveStyle(`color: red`);
  });
});
