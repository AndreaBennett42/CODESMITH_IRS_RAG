import { render, screen } from '@testing-library/react';
import { initI18n } from '../../i18n.js';
import DFAlert from './DFAlert.js';
import DFAccordion from '../Accordion/DFAccordion.js';

const i18next = await initI18n();
i18next.addResourceBundle(`test`, `translation`, {
  info: {
    '/info/path/to/alert': {
      alertText: {
        heading: `Alert Heading`,
        body: [{ p: `Alert Body` }],
      },
    },
    '/info/path/to/alert/expandable': {
      alertText: {
        heading: `Alert Heading with child config`,
      },
    },
    '/info/path/to/alert/expandable-explainer': {
      heading: `Click to see more details`,
      body: `This is the expanded body text`,
    },
  },
  'standalone.key': `Standalone Text`,
});

i18next.changeLanguage(`test`);

describe(`DFAlert component`, () => {
  const i18nKeyLongPath = `/info/path/to/alert`;
  const i18nKeyStandalone = `standalone.key`;

  it(`renders both heading and body when given a long path key`, () => {
    render(<DFAlert collectionId={null} i18nKey={i18nKeyLongPath} type='warning' headingLevel='h3' />);

    const headingElement = screen.getByText(/Alert Heading/);
    expect(headingElement).toBeInTheDocument();
    expect(headingElement).toHaveClass(`usa-alert__heading`);
    expect(headingElement.tagName).toBe(`H3`);
    expect(headingElement.closest(`.usa-alert`)).toHaveClass(`usa-alert--warning`);

    const bodyElement = screen.getByText(/Alert Body/);
    expect(bodyElement).toBeInTheDocument();
  });

  it(`renders standalone key as body by default`, () => {
    render(<DFAlert collectionId={null} i18nKey={i18nKeyStandalone} type='warning' headingLevel='h3' />);

    const bodyElement = screen.getByText(/Standalone Text/);
    expect(bodyElement).toBeInTheDocument();
  });

  it(`renders standalone key as heading when showTextAsHeader is true`, () => {
    render(
      <DFAlert
        collectionId={null}
        i18nKey={i18nKeyStandalone}
        showTextAsHeader={true}
        type='warning'
        headingLevel='h3'
      />
    );

    const headingElement = screen.getByText(/Standalone Text/);
    expect(headingElement).toBeInTheDocument();
    expect(headingElement).toHaveClass(`usa-alert__heading`);
  });

  it(`renders child configs`, () => {
    const i18nKeyWithExpandable = `/info/path/to/alert/expandable`;

    render(
      <DFAlert collectionId={null} i18nKey={i18nKeyWithExpandable} type='warning' headingLevel='h3'>
        <DFAccordion i18nKey={i18nKeyWithExpandable + `-explainer`} />
      </DFAlert>
    );

    expect(screen.getByText(/Alert Heading with child config/)).toBeInTheDocument();

    expect(screen.getByText(/Click to see more details/)).toBeInTheDocument();
  });
});
