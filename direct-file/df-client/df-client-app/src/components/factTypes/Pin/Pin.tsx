import { forwardRef, useState } from 'react';
import * as sfg from '@irs/js-factgraph-scala';
import { FactProps } from '../../../types/core.js';
import { useTranslation } from 'react-i18next';
import { stripNonNumeric } from '../../../misc/misc.js';
import useFact from '../../../hooks/useFact.js';
import { TextFormControl } from '../../FormControl/index.js';
import styles from './Pin.module.scss';

const Pin = forwardRef<HTMLInputElement, FactProps>(
  (
    { path, onValidData, showFeedback = false, isValid, required, concretePath, collectionId, hintKey, readOnly },
    ref
  ) => {
    const [fact, setFact, clearFact] = useFact<sfg.Pin>(concretePath);

    const { t } = useTranslation(`translation`);
    const [validationErrorCode, setValidationErrorCode] = useState<string>();

    const setValidity = (isValid: boolean) => {
      onValidData(concretePath, isValid);
    };

    const showError = showFeedback && !isValid;

    return (
      <TextFormControl
        path={path}
        collectionId={collectionId}
        concretePath={concretePath}
        hintKey={hintKey}
        setValidity={setValidity}
        onError={(err: sfg.PinValidationFailure) => {
          setValidationErrorCode(err.validationMessage.toUserFriendlyReason().toString());
        }}
        setFact={setFact}
        clearFact={clearFact}
        required={required}
        factParser={(rawValue: string) => sfg.PinFactory(stripNonNumeric(rawValue))}
        errorMessage={showError ? (t(`fields.${path}.errorMessages.${validationErrorCode}`) as string) : undefined}
        inputRef={ref}
        inputMode='numeric'
        type='text'
        maxLength={5}
        defaultValue={`${fact ?? ``}`}
        className={styles.dfPinInput}
        disabled={readOnly}
      />
    );
  }
);

Pin.displayName = `Pin`;

export default Pin;
