import { SimpleTextFormControl } from './TextFormControl.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { i18n, renderWithTranslationProvider as render } from '../../test/test-utils.js';
import { Path } from '../../fact-dictionary/Path.js';

const sharedProps = {
  path: `/fake/path` as Path,
  concretePath: `/fake/concrete-path` as unknown as ConcretePath,
  type: `text`,
  collectionId: null,
} as React.ComponentProps<typeof SimpleTextFormControl>;

const requiredMarker = `(Required)`;

const testConfig = {
  labelKey: `fields.${sharedProps.path}.name`,
  label: `test label`,
  hintKey: `info./info${sharedProps.path}.helpText.hint.text`,
  hint: `test hint`,
  hintIdSubstring: `__hint`,
  errorIdSubstring: `__error-msg`,
};

describe(`SimpleTextFormControl`, () => {
  beforeEach(() => {
    if (!i18n.exists(testConfig.labelKey)) {
      i18n.addResource(`en`, `test`, testConfig.labelKey, testConfig.label);
      i18n.addResource(`en`, `test`, `fields.generics.requiredExplainerSimple`, `Required`);
    }
  });

  function renderWrappedSimpleTextFormControl(
    overrides: Partial<React.PropsWithoutRef<typeof SimpleTextFormControl>> = {}
  ) {
    return render(<SimpleTextFormControl {...{ ...sharedProps, ...overrides }} />);
  }

  it(`renders without errors`, () => {
    const { getByRole } = renderWrappedSimpleTextFormControl();

    const textbox = getByRole(`textbox`);
    expect(textbox).not.toHaveAttribute(`aria-describedby`, expect.stringContaining(testConfig.hintIdSubstring));
    expect(textbox).not.toHaveAttribute(`aria-describedby`, expect.stringContaining(testConfig.errorIdSubstring));
  });

  it(`displays and translates label + required marker [default]`, () => {
    const { getByLabelText } = renderWrappedSimpleTextFormControl();

    expect(getByLabelText(`${i18n.t(testConfig.label)} ${requiredMarker}`)).toBeInTheDocument();
  });

  it(`displays and translates label [required = false]`, () => {
    const { getByLabelText } = renderWrappedSimpleTextFormControl({ required: false });

    expect(getByLabelText(`${i18n.t(testConfig.label)}`)).toBeInTheDocument();
  });

  it(`when there is no error, aria-describedby does not contain an errorId`, () => {
    const { getByRole } = renderWrappedSimpleTextFormControl();

    expect(getByRole(`textbox`)).not.toHaveAttribute(`aria-describedby`);
  });

  it(`when there is an error, aria-describedby contains an errorId`, () => {
    const errorMessage = `Something went wrong`;
    const { getByText, getByRole } = renderWrappedSimpleTextFormControl({ errorMessage });

    const errorMessageNode = getByText(errorMessage);
    expect(errorMessageNode).toBeInTheDocument();
    expect(getByRole(`textbox`)).toHaveAttribute(`aria-describedby`, expect.stringContaining(errorMessageNode.id));
  });

  it(`throws an error if mask and prefix are both set`, () => {
    expect(() => renderWrappedSimpleTextFormControl({ mask: `___-___-___`, inputPrefix: `$` })).toThrowError();
  });

  it(`throws an error if mask and suffix are both set`, () => {
    expect(() => renderWrappedSimpleTextFormControl({ mask: `___-___-___`, inputSuffix: `%` })).toThrowError();
  });

  describe(`with hint`, () => {
    beforeEach(() => {
      if (!i18n.exists(testConfig.labelKey)) {
        i18n.addResource(`en`, `test`, testConfig.labelKey, testConfig.label);
      }
      if (!i18n.exists(testConfig.hintKey)) {
        i18n.addResource(`en`, `test`, testConfig.hintKey, testConfig.hint);
      }
    });

    it(`input aria-describedby references hint`, () => {
      const { getByText, getByRole } = renderWrappedSimpleTextFormControl();

      const hintNode = getByText(testConfig.hint);
      expect(getByRole(`textbox`)).toHaveAttribute(`aria-describedby`, expect.stringContaining(hintNode.id));
    });

    it(`input aria-describedby references hint and error`, () => {
      const errorMessage = `test error`;
      const { getByText, getByRole } = renderWrappedSimpleTextFormControl({ errorMessage, isSensitive: false });

      const hintNode = getByText(testConfig.hint);
      const errorMessageNode = getByText(errorMessage);
      const textbox = getByRole(`textbox`);
      expect(textbox).toHaveAttribute(`aria-describedby`, expect.stringContaining(hintNode.id));
      expect(textbox).toHaveAttribute(`aria-describedby`, expect.stringContaining(errorMessageNode.id));
    });
  });

  afterEach(() => {
    i18n.removeResourceBundle(`en`, `test`);
  });
});
