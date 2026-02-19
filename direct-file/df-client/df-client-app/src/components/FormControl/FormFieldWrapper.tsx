import {
  FormFieldWrapperWithConfigurableLabel,
  FormFieldWrapperWithConfigurableLabelProps,
} from './FormFieldWrapperWithConfigurableLabel.js';
import { FormFieldLabel, FormFieldLabelProps } from './FormFieldLabel.js';

export function FormFieldWrapper({
  labelKey,
  labelContext,
  path,
  hintKey,
  name,
  labelId,
  collectionId,
  controlId,
  hintId,
  showError,
  errorMessage,
  useLegendAsLabel,
  children,
  readonly,
  required,
}: Omit<FormFieldWrapperWithConfigurableLabelProps, 'label'> & FormFieldLabelProps) {
  const label = (
    <FormFieldLabel
      labelKey={labelKey}
      labelContext={labelContext}
      labelId={labelId}
      collectionId={collectionId}
      controlId={controlId}
      showError={showError}
      useLegendAsLabel={useLegendAsLabel}
      readonly={readonly}
      required={required}
    />
  );

  return (
    <FormFieldWrapperWithConfigurableLabel
      path={path}
      hintKey={hintKey}
      name={name}
      collectionId={collectionId}
      controlId={controlId}
      hintId={hintId}
      showError={showError}
      errorMessage={errorMessage}
      readonly={readonly}
      label={label}
    >
      {children}
    </FormFieldWrapperWithConfigurableLabel>
  );
}
