import { JSeither } from '@irs/js-factgraph-scala';
import { useCallback, useState, Dispatch, SetStateAction } from 'react';

export type UseFactControlParameters<FactValue, FactError, RawValue> = {
  setValidity: (isValid: boolean) => void;
  setFact: (value: FactValue) => void;
  clearFact: () => void;
  isFactRequired: boolean;
  setInputValue?: Dispatch<SetStateAction<RawValue>>;
  sanitize?: (value: RawValue) => { value: RawValue; isEmpty: boolean };
  onError: (error: FactError) => void;
  factParser: (rawValue: RawValue) => JSeither<FactError, FactValue>;
};

/**
 * useFactControl manages the validation and state of a generic fact.
 * It also handles the optionality of a fact.
 *
 * @param setValidity Callback which sets the validity of the fact based on the result of fact validation
 * @param setFact Callback which sets the value of the fact
 * @param clearFact Callback which clears the value of the fact (sets to incomplete)
 * @param isFactRequired Boolean indicating whether the fact is required
 * @param onError Callback which handles the error in case of fact parsing failure
 * @param factParser Function which parses the raw input value into a scala result
 * @param sanitize Optional callback which sanitizes the raw input value
 * @param setInputValue Optional callback which sets a controlled input value
 *
 * @returns An object containing the `onChange` handler and the `rawValue` state.
 */
export function useFactControl<FactValue, FactError, RawValue>({
  setValidity,
  setFact,
  clearFact,
  isFactRequired = true,
  onError,
  factParser,
  sanitize,
  setInputValue,
}: UseFactControlParameters<FactValue, FactError, RawValue>) {
  const [rawValue, setRawValue] = useState<RawValue>();

  const onChange = useCallback(
    (rawValue: RawValue) => {
      setRawValue(rawValue);
      // Sanitized returns the value and whether it is to be considered "empty"
      // If a sanitizing function wasn't provided, return original value and consider non-empty
      // as that will handoff to the fact parser to handle validation
      const sanitized = sanitize ? sanitize(rawValue) : { value: rawValue, isEmpty: false };
      setInputValue?.(sanitized.value);

      if (sanitized.isEmpty) {
        // If optional and empty, bypass fact validation and clear the fact.
        // This makes it incomplete, but we still set to valid.
        clearFact();
        setValidity(!isFactRequired);
      } else {
        // Else, validate and set the fact, or return an error if not
        const result = factParser(sanitized.value);

        result.mapLeftRight(
          (err) => {
            onError(err);
            setValidity(false);
          },
          (value) => {
            setFact(value);
            setValidity(true);
          }
        );
      }
    },
    [factParser, onError, setFact, clearFact, isFactRequired, sanitize, setValidity, setInputValue]
  );

  return {
    onChange,
    rawValue,
  };
}
