import { forwardRef, useState } from 'react';
import * as sfg from '@irs/js-factgraph-scala';
import { FactProps } from '../../../types/core.js';
import { useTranslation } from 'react-i18next';
import { stripNonNumeric } from '../../../misc/misc.js';
import useFact from '../../../hooks/useFact.js';
import { TextFormControl } from '../../FormControl/index.js';
import DFModal from '../../HelperText/DFModal.js';

const IpPin = forwardRef<HTMLInputElement, FactProps>((props, ref) => {
  const { path, onValidData, showFeedback, isValid, concretePath, collectionId, required, readOnly } = props;
  const [fact, setFact, clearFact] = useFact<sfg.IpPin>(concretePath);

  const { t } = useTranslation(`translation`);
  const [validationErrorCode, setValidationErrorCode] = useState<string>();

  const setValidity = (isValid: boolean) => {
    onValidData(concretePath, isValid);
  };

  const showError = showFeedback && !isValid;

  return (
    <>
      {readOnly && (
        <DFModal
          batches={[`data-import-0`]}
          i18nKey='/info/you-and-your-family/about-you/why-cant-i-change-ip-pin'
          collectionId={collectionId}
        />
      )}
      <TextFormControl
        path={path}
        collectionId={collectionId}
        concretePath={concretePath}
        setValidity={setValidity}
        onError={(err: sfg.IpPinValidationFailure) => {
          setValidationErrorCode(err.validationMessage.toUserFriendlyReason().toString());
        }}
        setFact={setFact}
        clearFact={clearFact}
        required={required}
        factParser={(rawValue: string) => sfg.IpPinFactory(stripNonNumeric(rawValue))}
        errorMessage={showError ? (t(`fields.${path}.errorMessages.${validationErrorCode}`) as string) : undefined}
        inputRef={ref}
        inputMode='numeric'
        type='text'
        maxLength={6}
        readOnly={readOnly}
        defaultValue={`${fact ?? ``}`}
      />
    </>
  );
});

IpPin.displayName = `IpPin`;

export default IpPin;
