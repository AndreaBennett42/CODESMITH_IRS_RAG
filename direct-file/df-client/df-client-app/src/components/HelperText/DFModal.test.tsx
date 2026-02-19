/* eslint-disable max-len */
import { render, screen } from '@testing-library/react';
import { initI18n } from '../../i18n.js';
import DFModal from './DFModal.js';
import { ItemConfig } from '../ConditionalList/ConditionalList.js';

const i18next = await initI18n();
i18next.addResourceBundle(`test`, `translation`, {
  modals: {
    sharedModalW2: {
      header: `What's this?`,
      body: `We must also be in touch with the <Link1>wonders of life</Link1>. They are within us and all around us, everywhere, all the time.`,
      urls: {
        Link1: `https://www.example.com/linkmodal2`,
      },
    },
  },
  info: {
    '/info/you-and-your-family/you/intro': {
      helpText: {
        modals: {
          text: [
            {
              p: `Tell us about yourself. <LinkModal1>More in the</LinkModal1>`,
            },
            {
              p: `Why do we <LinkModal2>ask for contact</LinkModal2> <LinkModal3>information?</LinkModal3>`,
            },
            {
              p: `Sharing is <sharedModalW2>caring</sharedModalW2>.`,
            },
          ],
          LinkModal1: {
            header: `Let me explain`,
            body: [
              {
                p: `Your mailing address is only used if the IRS needs to contact you <Link1>or to mail</Link1> your tax refund.`,
              },
              {
                p: `If the IRS needs to contact you, they'll <Link2>send you one</Link2> or more notices by US mail. The IRS only calls by phone in special cases.`,
              },
            ],
            urls: {
              Link1: `https://www.irs.gov/newsroom/avoid-scams-know-the-facts-on-how-the-irs-contacts-taxpayers`,
              Link2: `https://www.example.com`,
            },
          },
          LinkModal2: {
            header: `What's this?`,
            body: `This is the explanation for <Link1>our second modal link</Link1>.`,
            urls: {
              Link1: `https://www.example.com/linkmodal2`,
            },
          },
          LinkModal3: {
            header: `Conditional Modal`,
            conditionOne: {
              body: `This is the first condition.`,
            },
            conditionTwo: {
              body: `This is the second condition.`,
            },
          },
        },
      },
    },
  },
  headings: {
    '/heading/test/intro': {
      helpText: {
        modals: {
          text: `<LinkModal1>What information do we need?</LinkModal1>`,
          LinkModal1: {
            header: `Not that much!`,
            body: `Short body text`,
          },
        },
      },
    },
  },
});
i18next.changeLanguage(`test`);

describe(`DFModal`, () => {
  const i18nKey = `/info/you-and-your-family/you/intro`;
  const headingKey = `/heading/test/intro`;

  it(`renders text with a modal link`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    expect(screen.getByText(/More in the/)).toBeInTheDocument();
  });

  it(`renders text with a modal link in the headings namespace`, () => {
    render(<DFModal collectionId={null} i18nKey={headingKey} />);
    expect(screen.getByText(/What information do we need?/)).toBeInTheDocument();
  });

  it(`renders the text content inside body`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    expect(screen.getByText(/Your mailing address is only used if the IRS needs to contact you/)).toBeInTheDocument();
    expect(screen.getByText(/The IRS only calls by phone in special cases./)).toBeInTheDocument();
  });

  it(`renders the first Link1 correctly`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    const link1Text = screen.getByText(/More in the/);
    expect(link1Text).toHaveAttribute(`href`, `#info`);
  });

  it(`renders the second Link1 correctly`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    const link1Body = screen.getByText(/or to mail/);
    expect(link1Body).toHaveAttribute(
      `href`,
      `https://www.irs.gov/newsroom/avoid-scams-know-the-facts-on-how-the-irs-contacts-taxpayers`
    );
  });

  it(`renders the Link2 correctly`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    const link2 = screen.getByText(/send you one/);
    expect(link2).toHaveAttribute(`href`, `https://www.example.com`);
  });

  it(`renders the LinkModal2 correctly`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    const linkModal2Text = screen.getByText(/ask for contact/);
    expect(linkModal2Text).toBeInTheDocument();
  });

  it(`renders the Link inside LinkModal2 correctly`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    const link3 = screen.getByText(/our second modal link/);
    expect(link3).toHaveAttribute(`href`, `https://www.example.com/linkmodal2`);
  });

  it(`renders the LinkModal3 correctly with conditional sections`, () => {
    const items = [{ itemKey: `conditionOne` }] as ItemConfig[];
    render(<DFModal collectionId={null} i18nKey={i18nKey} items={items} />);
    const linkModal3Text = screen.getByText(/ask for contact/);
    expect(linkModal3Text).toBeInTheDocument();
    expect(screen.getByText(/This is the first condition./)).toBeInTheDocument();
    expect(screen.queryByText(/This is the second condition./)).not.toBeInTheDocument();
  });

  it(`renders the sharedModalW2 correctly`, () => {
    render(<DFModal collectionId={null} i18nKey={i18nKey} />);
    expect(screen.getByRole(`button`, { name: `caring` })).toBeInTheDocument();
    expect(screen.getByRole(`link`, { name: `wonders of life` })).toBeInTheDocument();
    expect(screen.getByText(/within us and all around us/)).toBeInTheDocument();
  });
});
