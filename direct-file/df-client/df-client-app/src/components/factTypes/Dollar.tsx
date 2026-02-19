import * as sfg from '@irs/js-factgraph-scala';
import { forwardRef, useState, useCallback, useRef, useEffect } from 'react';

import { type Dollar, type DollarValidationFailure } from '@irs/js-factgraph-scala';

import { type FactProps } from '../../types/core.js';
import { LabelConfig, TextFormControl } from '../FormControl/index.js';
import useFact from '../../hooks/useFact.js';
import Translation from '../Translation/index.js';
import { getFactFloatLimit } from '../../factgraph/factLimitHelpers.js';

/**
 * Strip out any always disallowed characters
 * @param rawValue
 */
function sanitizeDollar(rawValue: string) {
  // Strip out anything aside from numbers, commas, or decimals
  const sanitized = rawValue.replace(/[^-0-9,.()]/g, ``);

  return { value: sanitized, isEmpty: !sanitized };
}

export interface DollarImplementationProps extends FactProps {
  fact: Dollar | undefined;
  setFact: (value: Dollar) => void;
  clearFact: () => void;
  labelConfig?: LabelConfig;
  controlIdOverride?: string;
  forceUpdateInput?: boolean;
}

/**
 * The underlying Dollar implementation, decoupled from the fact graph
 */
export const DollarImplementation = forwardRef<HTMLInputElement, DollarImplementationProps>(
  (
    {
      path,
      onValidData,
      showFeedback = false,
      isValid,
      required,
      concretePath,
      collectionId,
      hintKey,
      fact,
      setFact,
      clearFact,
      controlIdOverride,
      forceUpdateInput,
      readOnly,
      labelConfig,
    },
    ref
  ) => {
    const [inputValue, setInputValue] = useState(`${fact || ``}`);
    const [validationErrorCode, setValidationErrorCode] = useState<string>();

    const prevFactValue = useRef(fact);

    useEffect(() => {
      // We have to track and handle if an update occurs from underneath of us
      // Normally in react we would handle this via state boosting, but this
      // state only lives in the fact graph

      // This check exists here specifically to account for ImportAgiContent,
      // but may need to expand to non-dollar inputs in the future.
      if (forceUpdateInput && fact !== prevFactValue) {
        setInputValue(`${fact || ``}`);
        // We have to note that it's valid so we don't get caught by the "field is required" check
        onValidData(concretePath, true);
        prevFactValue.current = fact;
      }
    }, [fact, prevFactValue, setInputValue, concretePath, onValidData, forceUpdateInput]);

    const setValidity = (isValid: boolean) => {
      if (isValid && validationErrorCode !== undefined) {
        setValidationErrorCode(undefined);
      }
      onValidData(concretePath, isValid);
    };
    const onError = (error: DollarValidationFailure) => {
      setValidationErrorCode(error.validationMessage.toUserFriendlyReason().toString());
    };
    const factParser = useCallback(
      (rawValue: string) => {
        const maxLimit = getFactFloatLimit(path, `Max`);
        const minLimit = getFactFloatLimit(path, `Min`);
        return sfg.DollarFactory(rawValue, maxLimit, minLimit);
      },
      [path]
    );

    const showError = showFeedback && !isValid;

    const errorMessage = showError ? (
      <Translation
        i18nKey={[
          // Use custom message if available, fallback to global for type
          `fields.${path}.errorMessages.${validationErrorCode}`,
          `fields.generics.dollar.errorMessages.${validationErrorCode}`,
        ]}
        collectionId={collectionId}
        context={{
          maximumValue: getFactFloatLimit(path, `Max`),
          minimumValue: getFactFloatLimit(path, `Min`),
        }}
      />
    ) : undefined;

    return (
      <TextFormControl
        labelConfig={labelConfig}
        inputPrefix='$'
        path={path}
        concretePath={concretePath}
        collectionId={collectionId}
        hintKey={hintKey}
        setValidity={setValidity}
        setFact={setFact}
        clearFact={clearFact}
        required={required}
        setInputValue={setInputValue}
        onError={onError}
        sanitize={sanitizeDollar}
        factParser={factParser}
        type='text'
        inputMode='decimal'
        errorMessage={errorMessage}
        inputRef={ref}
        controlIdOverride={controlIdOverride}
        // We need this to be a controlled component in order to prevent any invalid characters
        value={inputValue}
        disabled={readOnly}
      />
    );
  }
);
DollarImplementation.displayName = `DollarImplementation`;

/**
 * Flow-ready control for Dollar facts
 */
const Dollar = forwardRef<HTMLInputElement, FactProps>(({ concretePath, ...props }, ref) => {
  const [fact, setFact, clearFact] = useFact<Dollar>(concretePath);

  return (
    <DollarImplementation
      concretePath={concretePath}
      fact={fact}
      setFact={setFact}
      clearFact={clearFact}
      forceUpdateInput
      {...props}
      ref={ref}
    />
  );
});
Dollar.displayName = `Dollar`;

export default Dollar;
