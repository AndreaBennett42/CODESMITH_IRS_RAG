import { useState, forwardRef } from 'react';
import * as sfg from '@irs/js-factgraph-scala';
import { FactProps } from '../../../types/core.js';
import { useTranslation } from 'react-i18next';
import { stripNonNumeric } from '../../../misc/misc.js';
import useFact from '../../../hooks/useFact.js';
import { TextFormControl } from '../../FormControl/index.js';

const Ein = forwardRef<HTMLInputElement, FactProps>(
  (
    { path, onValidData, showFeedback = false, isValid, required, concretePath, readOnly, collectionId, hintKey },
    ref
  ) => {
    const [fact, setFact, clearFact] = useFact<sfg.Ein>(concretePath);

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
        onError={(err: sfg.EinValidationFailure) => {
          setValidationErrorCode(err.validationMessage.toUserFriendlyReason().toString());
        }}
        setFact={setFact}
        clearFact={clearFact}
        required={required}
        factParser={(rawValue: string) => sfg.EinFactory(stripNonNumeric(rawValue))}
        errorMessage={showError ? (t(`fields.${path}.errorMessages.${validationErrorCode}`) as string) : undefined}
        inputMode='numeric'
        type='text'
        mask={`__-_______`}
        readOnly={readOnly}
        defaultValue={`${fact ?? ``}`}
        inputRef={ref}
      />
    );
  }
);

Ein.displayName = `Ein`;

export default Ein;
