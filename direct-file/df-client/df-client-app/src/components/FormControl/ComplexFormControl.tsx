import { ConcretePath } from '@irs/js-factgraph-scala';
import { Fieldset, FormGroup, Label } from '@trussworks/react-uswds';
import classNames from 'classnames';
import Translation from '../Translation/index.js';
import { PAGE_HEADING_ID } from '../PageTitle/index.js';
import { QuietErrorMessage } from '../QuietErrorMessage.js';
import { buildFormControlId } from './helpers.js';
import RequiredMarker from './RequiredMarker.js';
import { Path } from '../../fact-dictionary/Path.js';

export function ComplexFormControl({
  path,
  concretePath,
  showError,
  children,
  errorMessage,
  labelledBy = `legend`,
  hintId,
  className,
  required = true,
  shouldRenderAsSelect,
  labelFor,
}: {
  path: Path;
  concretePath: ConcretePath;
  showError: boolean;
  children: React.ReactElement | React.ReactElement[];
  errorMessage: string | JSX.Element;
  /**
   * - `heading`: programmatically label the field with the page's h1
   * - `legend`: use `<legend>` with the text found at `fields.${path}.name` in en.yaml
   * - `self`: children must already be labelled
   */
  labelledBy?: 'heading' | 'legend' | 'self';
  hintId?: string;
  className?: string;
  required?: boolean;
  shouldRenderAsSelect?: boolean;
  labelFor?: string;
}) {
  const legendStyle = classNames({ 'usa-label--error': showError });

  const controlId = buildFormControlId(concretePath);
  const errorId = showError ? `${controlId}__error-msg` : undefined;

  const additionalFieldsetProps = labelledBy === `heading` ? { [`aria-labelledby`]: PAGE_HEADING_ID } : {};
  const showStandAloneRequiredMarker = required && labelledBy === `heading`;

  const ids = [];
  if (showError) {
    ids.push(errorId);
  }
  if (hintId) {
    ids.push(hintId);
  }
  // NOTE: The aria-describedby only triggers if the user specifically navigates to the fieldset
  //    it will not be read if the user tabs to an inner component
  // TODO: Set this up so select boxes are not inside of <fieldset>
  return (
    <>
      <FormGroup error={showError} className={className}>
        {shouldRenderAsSelect ? (
          <>
            {showError && <QuietErrorMessage id={errorId}>{errorMessage}</QuietErrorMessage>}

            <Label htmlFor={labelFor ? labelFor : `${path}`}>
              <Translation i18nKey={`fields.${path}.name`} collectionId={null} />
              {` `}
              {required && <RequiredMarker />}
            </Label>
            {children}
          </>
        ) : (
          <>
            {showStandAloneRequiredMarker && <RequiredMarker inline={false} />}
            <Fieldset
              name={path}
              {...additionalFieldsetProps}
              id={controlId}
              aria-describedby={ids.length > 0 ? ids.join(` `) : undefined}
            >
              {labelledBy === `legend` && (
                <legend className={legendStyle} data-testid='legend'>
                  <Translation i18nKey={`fields.${path}.name`} collectionId={null} />
                  {` `}
                  {required && <RequiredMarker />}
                </legend>
              )}
              {showError && <QuietErrorMessage id={errorId}>{errorMessage}</QuietErrorMessage>}
              {children}
            </Fieldset>
          </>
        )}
      </FormGroup>
    </>
  );
}
