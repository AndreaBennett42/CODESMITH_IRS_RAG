import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { initI18n } from '../../i18n.js';
import { ConditionalList, ConditionalListProps } from './ConditionalList.js';
import { FactGraphContextProvider } from '../../factgraph/FactGraphContext.js';
import { setupStore } from '../../redux/store.js';
import {
  createBooleanWrapper,
  createCollectionWrapper,
  createEnumWrapper,
  createStringWrapper,
} from '../../test/persistenceWrappers.js';

const i18next = await initI18n();
i18next.addResourceBundle(`test`, `translation`, {
  info: {
    '/info/path/to/list': {
      firstName: `firstName`,
      keyMock1: `Key 1`,
      keyMock2: `Key 2`,
      keyMock3: `Key 3`,
      keyMock4: {
        helpText: {
          modals: {
            text: `<LinkModal1>What if I changed</LinkModal1> my legal name last tax year?`,
            LinkModal1: {
              header: `If you changed your name last tax year`,
              body: {
                text: `You must change it back`,
              },
            },
          },
        },
      },
    },
    '/info/path/to/prefix': `Prefix to list:`,
  },
  'standalone.key': `Standalone Text`,
});

i18next.changeLanguage(`test`);

// the mock fact has to be a writable boolean; not a derived fact
const props: ConditionalListProps = {
  items: [
    {
      itemKey: `keyMock1`,
      conditions: [`/wantsJointReturn`],
    },
    {
      itemKey: `keyMock2`,
      conditions: [{ operator: `isFalse`, condition: `/wantsJointReturn` }],
    },
    {
      itemKey: `keyMock3`,
      conditions: [`/wantsJointReturn`, `/spouseItemizes`],
    },
    {
      itemKey: `keyMock4`,
      conditions: undefined,
    },
  ],
  collectionId: null,
  i18nKey: `info./info/path/to/list`,
  i18nPrefixKey: `info./info/path/to/prefix`,
};

describe(`ConditionalList component`, () => {
  it(`renders the wants-true item when the condition is true`, () => {
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider
          forceNewInstance
          existingFacts={{
            '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
          }}
        >
          <ConditionalList {...props} />
        </FactGraphContextProvider>
      </Provider>
    );
    expect(screen.getByText(/Key 1/)).toBeInTheDocument();
  });

  it(`renders the wants-false item and not the wants-true item when the condition is false`, () => {
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider
          forceNewInstance
          existingFacts={{
            '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
          }}
        >
          <ConditionalList {...props} />
        </FactGraphContextProvider>
      </Provider>
    );
    expect(screen.queryByText(/Key 1/)).not.toBeInTheDocument();
    expect(screen.getByText(/Key 2/)).toBeInTheDocument();
  });

  it(`won't render a key if any of its conditions are false`, () => {
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider
          forceNewInstance
          existingFacts={{
            '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
            '/spouseItemizes': createBooleanWrapper(false),
          }}
        >
          <ConditionalList {...props} />
        </FactGraphContextProvider>
      </Provider>
    );
    expect(screen.queryByText(/Key 3/)).not.toBeInTheDocument();
  });

  it(`will render a key when all of its conditions are true`, () => {
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider
          forceNewInstance
          existingFacts={{
            '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
            '/spouseItemizes': createBooleanWrapper(true),
          }}
        >
          <ConditionalList {...props} />
        </FactGraphContextProvider>
      </Provider>
    );
    expect(screen.queryByText(/Key 3/)).toBeInTheDocument();
  });

  it(`renders items as modals when the input describes a modal`, () => {
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider>
          <ConditionalList {...props} />
        </FactGraphContextProvider>
      </Provider>
    );
    const link1Text = screen.getByText(/What if I changed/);
    expect(link1Text).toHaveAttribute(`href`, `#info`);
  });

  it(`a key renders a collection when passed a collection`, () => {
    const uuid1 = `159c03d1-af4a-447f-96aa-d19397048a44`;
    const uuid2 = `259c03d1-af4a-447f-96aa-d19397048a44`;
    const facts = {
      '/filers': createCollectionWrapper([uuid1, uuid2]),
      [`/filers/#${uuid1}/firstName`]: createStringWrapper(`Alex`),
      [`/filers/#${uuid2}/firstName`]: createStringWrapper(`Mark`),
    };
    const listProps: ConditionalListProps = {
      items: [
        {
          itemKey: `firstName`,
          collection: `/filers`,
        },
        {
          itemKey: `keyMock2`,
        },
      ],
      collectionId: null,
      i18nKey: `info./info/path/to/list`,
    };
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider forceNewInstance existingFacts={facts}>
          <ConditionalList {...listProps} />
        </FactGraphContextProvider>
      </Provider>
    );
    // We haven't set up string interpolation in the test, so
    // the actual first names are not subbed in for testing -- instead
    // we just look for the key "firstName"
    expect(screen.getAllByText(/firstName/)).toHaveLength(2);
    expect(screen.getByText(/Key 2/)).toBeInTheDocument();
  });

  it(`renders the prefix when passed a prefix key`, () => {
    const uuid1 = `159c03d1-af4a-447f-96aa-d19397048a44`;
    const facts = {
      '/filers': createCollectionWrapper([uuid1]),
      [`/filers/#${uuid1}/firstName`]: createStringWrapper(`bell`),
    };
    const listProps: ConditionalListProps = {
      items: [],
      collectionId: null,
      i18nKey: `info./info/path/to/list`,
      i18nPrefixKey: `info./info/path/to/prefix`,
    };
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider forceNewInstance existingFacts={facts}>
          <ConditionalList {...listProps} />
        </FactGraphContextProvider>
      </Provider>
    );

    expect(screen.getByText(`info./info/path/to/prefix.body`)).toBeInTheDocument();
  });

  it(`does not render the prefix when not passed a prefix key`, () => {
    const uuid1 = `159c03d1-af4a-447f-96aa-d19397048a44`;
    const facts = {
      '/filers': createCollectionWrapper([uuid1]),
      [`/filers/#${uuid1}/firstName`]: createStringWrapper(`hooks`),
    };
    const listProps: ConditionalListProps = {
      items: [],
      collectionId: null,
      i18nKey: `info./info/path/to/list`,
    };
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider forceNewInstance existingFacts={facts}>
          <ConditionalList {...listProps} />
        </FactGraphContextProvider>
      </Provider>
    );

    expect(screen.queryByText(`info./info/path/to/prefix.body`)).not.toBeInTheDocument();
  });

  it(`renders as a summary box if isSummary is true`, () => {
    const summaryBoxProp = { ...props, isSummary: true };
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider>
          <ConditionalList {...summaryBoxProp} />
        </FactGraphContextProvider>
      </Provider>
    );
    const summaryBox = screen.queryByTestId(`conditional-list-summary-box`);
    expect(summaryBox).toBeInTheDocument();
  });
  it(`does NOT render as a summary box if isSummary is false`, () => {
    const summaryBoxProp = { ...props, isSummary: false };
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider>
          <ConditionalList {...summaryBoxProp} />
        </FactGraphContextProvider>
      </Provider>
    );
    const summaryBox = screen.queryByTestId(`conditional-list-summary-box`);
    expect(summaryBox).toBeNull();
  });
  it(`does NOT render as a summary box if isSummary is not specified`, () => {
    render(
      <Provider store={setupStore()}>
        <FactGraphContextProvider>
          <ConditionalList {...props} />
        </FactGraphContextProvider>
      </Provider>
    );
    const summaryBox = screen.queryByTestId(`conditional-list-summary-box`);
    expect(summaryBox).toBeNull();
  });
});
