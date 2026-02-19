import { Label } from '@trussworks/react-uswds';
import Translation from '../Translation/index.js';
import classNames from 'classnames';
import RequiredMarker from './RequiredMarker.js';
import { I18nKey } from '../Translation/Translation.js';

export interface FormFieldLabelProps {
  labelKey: I18nKey;
  labelContext?: object;
  labelId?: string;
  collectionId: string | null;
  controlId: string;
  showError: boolean;
  useLegendAsLabel?: boolean;
  readonly?: boolean;
  required?: boolean;
}

export function FormFieldLabel({
  labelKey,
  labelContext,
  labelId,
  collectionId,
  controlId,
  showError,
  useLegendAsLabel,
  readonly,
  required,
}: FormFieldLabelProps) {
  const labelBody = <Translation i18nKey={labelKey} collectionId={collectionId} context={labelContext} />;
  const legendStyle = classNames({ 'usa-label--error': showError, 'float-left margin-right-05': useLegendAsLabel });
  const isRequired = required === false ? false : true;

  if (useLegendAsLabel) {
    return (
      <>
        <legend id={labelId} className={legendStyle} data-testid='legend'>
          {labelBody}
        </legend>
        {isRequired && !readonly && <RequiredMarker />}
      </>
    );
  }

  return (
    <Label id={labelId} htmlFor={controlId} error={showError}>
      {labelBody}
      {` `}
      {isRequired && !readonly && <RequiredMarker />}
    </Label>
  );
}
