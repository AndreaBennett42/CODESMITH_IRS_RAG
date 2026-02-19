import { render, screen } from '@testing-library/react';
import { initI18n } from '../../i18n.js';
import Hint from './Hint.js';

const i18next = await initI18n();
i18next.addResourceBundle(`test`, `translation`, {
  info: {
    '/info/path/to/hint': {
      body: `Info body`,
      helpText: {
        hint: {
          text: `This is <Link1>a hint</Link1>`,
          urls: {
            Link1: `https://www.example.com`,
          },
        },
      },
    },
  },
});
i18next.changeLanguage(`test`);

describe(`hint helper text`, () => {
  const i18nKeyLongPath = `/path/to/hint`;
  it(`renders hint helper text`, () => {
    render(<Hint hintId='hint-1234' collectionId={null} i18nKey={i18nKeyLongPath} />);
    expect(screen.getByText(/This is/)).toBeInTheDocument();
  });

  it(`hint helper text has class usa-hint`, () => {
    render(<Hint hintId='hint-1234' collectionId={null} i18nKey={i18nKeyLongPath} />);
    expect(screen.getByText(/This is/)).toHaveClass(`usa-hint`);
  });

  it(`renders the Link1 correctly in the hint`, () => {
    render(<Hint hintId='hint-1234' collectionId={null} i18nKey={i18nKeyLongPath} />);
    const link1 = screen.getByText(/a hint/);
    expect(link1).toHaveAttribute(`href`, `https://www.example.com`);
  });
});
