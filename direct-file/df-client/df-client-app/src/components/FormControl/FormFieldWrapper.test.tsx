import { i18n, renderWithTranslationProvider as render } from '../../test/test-utils.js';
import { FormFieldWrapper } from './FormFieldWrapper.js';
import { buildHintKey, buildReadonlyHintKey } from '../FormControl/helpers.js';
import { Path } from '../../fact-dictionary/Path.js';

const defaultTestConfig: React.ComponentProps<typeof FormFieldWrapper> = {
  labelKey: `test.label.key`,
  collectionId: `test-collection-id`,
  controlId: `test-control-id`,
  path: `/test/path` as Path,
  showError: false,
  children: <></>,
};
i18n.addResource(`en`, `test`, defaultTestConfig.labelKey as string, `test label`);

const renderFormFieldWrapper = (testConfig: Partial<Parameters<typeof FormFieldWrapper>[0]> = {}) =>
  render(<FormFieldWrapper {...{ ...defaultTestConfig, ...testConfig }} />);

describe(`FormFieldWrapper`, () => {
  it(`renders without errors with default test configuration`, () => {
    const { getByTestId, getByText } = renderFormFieldWrapper();

    expect(getByTestId(`label`)).toBeInTheDocument();
    expect(getByText(i18n.t(defaultTestConfig.labelKey))).toBeInTheDocument();
  });

  it(`displays error message`, () => {
    const testConfig = {
      showError: true,
      errorMessage: `Test error message`,
    };
    const { getByText } = renderFormFieldWrapper(testConfig);

    expect(getByText(testConfig.errorMessage)).toBeInTheDocument();
  });

  it(`displays hint`, () => {
    const testConfig = {
      hintKey: buildHintKey(defaultTestConfig.path as Path),
      hintId: `test.hint.id`,
    };

    //add translation since <Hint> will not render if key does not exist
    i18n.addResource(`en`, `test`, testConfig.hintKey, `test hint`);
    const { getByText } = renderFormFieldWrapper(testConfig);
    expect(getByText(i18n.t(testConfig.hintKey))).toBeInTheDocument();
  });

  it(`displays readonly hint`, () => {
    const testConfig = {
      hintKey: buildReadonlyHintKey(defaultTestConfig.path as Path),
      readonly: true,
    };
    //add translation since <DFModal> will not render if modal key does not exist
    i18n.addResourceBundle(`en`, `test`, {
      info: {
        '/test/path': {
          readOnlyField: {
            helpText: {
              modals: {
                text: `<LinkModal1>link</LinkModal1>`,
                LinkModal1: {
                  header: `test link`,
                  urls: {
                    Link1: `example.com`,
                  },
                  body: [{ p: `test body` }],
                },
              },
            },
          },
        },
      },
    });
    const { getByRole } = renderFormFieldWrapper(testConfig);
    const modal = getByRole(`dialog`);
    expect(modal).toHaveAttribute(`id`, `${testConfig.hintKey}-LinkModal1`);
  });

  it(`displays legend`, () => {
    const testConfig = {
      useLegendAsLabel: true,
    };
    const { getByTestId, getByText } = renderFormFieldWrapper(testConfig);

    expect(getByTestId(`legend`)).toBeInTheDocument();
    expect(getByText(i18n.t(defaultTestConfig.labelKey))).toBeInTheDocument();
  });

  afterAll(() => {
    i18n.removeResourceBundle(`en`, `test`);
  });
});
