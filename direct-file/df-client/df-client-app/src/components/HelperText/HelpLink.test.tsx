import { render, screen } from '@testing-library/react';
import { initI18n } from '../../i18n.js';
import HelpLink from './HelpLink.js';

const i18next = await initI18n();
i18next.addResourceBundle(`test`, `translation`, {
  info: {
    '/path/to/Link': {
      body: `Info body`,
      helpText: {
        helpLink: {
          text: `This <Link1>is a link to</Link1> an <Link2>external</Link2> site`,
          urls: {
            Link1: `https://www.example.com`,
            Link2: `https://www.irs.gov`,
          },
        },
      },
    },
  },
});
i18next.changeLanguage(`test`);

describe(`HelpLink`, () => {
  const i18nKey = `/path/to/Link`;

  it(`renders text with an external link`, () => {
    render(<HelpLink collectionId={null} i18nKey={i18nKey} gotoNextScreen={() => {}} />);
    expect(screen.getByText(/is a link to/)).toBeInTheDocument();
    expect(screen.getByText(/external/)).toBeInTheDocument();
  });

  it(`renders the Link1 correctly and checks for the external link class`, () => {
    render(<HelpLink collectionId={null} i18nKey={i18nKey} gotoNextScreen={() => {}} />);
    const link1 = screen.getByText(/is a link to/);
    expect(link1).toHaveAttribute(`href`, `https://www.example.com`);
  });

  it(`renders the Link2 correctly and checks for the external link class`, () => {
    render(<HelpLink collectionId={null} i18nKey={i18nKey} gotoNextScreen={() => {}} />);
    const link2 = screen.getByText(/external/);
    expect(link2).toHaveAttribute(`href`, `https://www.irs.gov`);
  });
});
