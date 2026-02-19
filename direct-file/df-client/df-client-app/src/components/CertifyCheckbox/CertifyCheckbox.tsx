import { useState } from 'react';
import { Checkbox, Fieldset, FormGroup } from '@trussworks/react-uswds';

import { InfoDisplayProps } from '../../types/core.js';
import Translation from '../Translation/index.js';
import { QuietErrorMessage } from '../QuietErrorMessage.js';
import RequiredMarker from '../FormControl/RequiredMarker.js';
import { checkFormControlValidity } from '../../misc/constraintsApi.js';

type CertifyCheckboxProps = { showFeedback?: boolean } & InfoDisplayProps;

const CertifyCheckbox = ({ i18nKey, showFeedback = false, handleChange }: CertifyCheckboxProps) => {
  const [isValid, setValidity] = useState(false);
  const showError = showFeedback && !isValid;
  const controlId = `id-${i18nKey}`;
  const errorId = showError ? `${controlId}__error-msg` : undefined;

  const internalHandleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValidity(checkFormControlValidity(`id-${i18nKey}-checkbox`));
    handleChange?.(e);
  };

  return (
    <FormGroup error={showError}>
      <RequiredMarker inline={false} />
      <Fieldset name={i18nKey} id={controlId} aria-describedby={errorId}>
        {showError && (
          <QuietErrorMessage id={errorId}>
            <Translation i18nKey={`enums.messages.requiredField`} collectionId={null} />
          </QuietErrorMessage>
        )}
        <Checkbox
          label={<Translation i18nKey={`fields.${i18nKey}.label`} collectionId={null} />}
          id={`id-${i18nKey}-checkbox`}
          name={`${i18nKey}-checkbox`}
          onChange={internalHandleChange}
          required
        />
      </Fieldset>
    </FormGroup>
  );
};

export default CertifyCheckbox;
