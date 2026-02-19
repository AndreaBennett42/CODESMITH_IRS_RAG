import * as sfg from '@irs/js-factgraph-scala';
import { forwardRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { FactProps } from '../../../types/core.js';
import useFact from '../../../hooks/useFact.js';
import { stripNonNumeric } from '../../../misc/misc.js';
import { TextFormControl } from '../../FormControl/index.js';

const PhoneNumber = forwardRef<HTMLInputElement, FactProps>(
  (
    {
      path,
      onValidData,
      showFeedback = false,
      isValid,
      concretePath,
      collectionId,
      required,
      hintKey,
      readOnly,
      autoComplete,
    },
    ref
  ) => {
    const [fact, setFact, clearFact] = useFact<sfg.UsPhoneNumber>(concretePath);

    const { t } = useTranslation(`translation`);
    const [validationErrorCode, setValidationErrorCode] = useState<string>();
    const setValidity = (isValid: boolean) => {
      onValidData(concretePath, isValid);
    };
    const showError = showFeedback && !isValid;

    return (
      <TextFormControl
        path={path}
        concretePath={concretePath}
        collectionId={collectionId}
        hintKey={hintKey}
        setValidity={setValidity}
        onError={(err: sfg.UsPhoneNumberValidationFailure) => {
          setValidationErrorCode(err.validationMessage.toUserFriendlyReason().toString());
        }}
        setFact={setFact}
        clearFact={clearFact}
        required={required}
        factParser={(rawValue: string) => sfg.UsPhoneNumberFactory(`+1${stripNonNumeric(rawValue)}`)}
        errorMessage={
          showError ? (t(`fields./phone.usPhoneNumber.errorMessages.${validationErrorCode}`) as string) : undefined
        }
        inputRef={ref}
        inputMode='numeric'
        type='tel'
        defaultValue={fact?.subscriberNumber ?? undefined}
        mask='___-___-____'
        autoComplete={autoComplete}
        readOnly={readOnly}
      />
    );
  }
);
PhoneNumber.displayName = `PhoneNumber`;

export default PhoneNumber;
