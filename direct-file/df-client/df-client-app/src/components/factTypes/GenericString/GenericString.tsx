import { useTranslation } from 'react-i18next';
import { forwardRef, useCallback } from 'react';
import { FactProps } from '../../../types/core.js';
import useFact from '../../../hooks/useFact.js';
import { SimpleTextFormControl, sanitizeString } from '../../FormControl/index.js';
import { getFactIntLimit } from '../../../factgraph/factLimitHelpers.js';

const GenericString = forwardRef<HTMLInputElement, FactProps>(
  (
    {
      path,
      onValidData,
      showFeedback = false,
      concretePath,
      isValid,
      readOnly,
      required = true,
      collectionId,
      hintKey,
      autoComplete,
    },
    ref
  ) => {
    const [fact, setFact, clearFact] = useFact<string>(concretePath);

    const { t } = useTranslation(`translation`);

    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const { value } = sanitizeString(e.target.value);
        if (readOnly !== true) {
          if (value.length > 0) {
            try {
              setFact(sanitizeString(value).value);
              onValidData(concretePath, true);
            } catch (e) {
              onValidData(concretePath, false);
            }
          } else if (!required) {
            // If the value is !required (aka optional) and empty, we clear the fact in the dictionary.
            // This makes it incomplete, but we still set to valid.
            clearFact();
            onValidData(concretePath, true);
          } else {
            onValidData(concretePath, false);
          }
        }
      },
      [clearFact, concretePath, onValidData, readOnly, required, setFact]
    );

    const showError = showFeedback && !isValid;

    return (
      <SimpleTextFormControl
        path={path}
        hintKey={hintKey}
        collectionId={collectionId}
        concretePath={concretePath}
        errorMessage={showError ? t(`enums.messages.requiredField`) : undefined}
        key={path}
        inputRef={ref}
        type='text'
        onChange={handleChange}
        required={required}
        defaultValue={fact ?? ``}
        readOnly={readOnly}
        maxLength={getFactIntLimit(path, `MaxLength`)}
        autoComplete={autoComplete}
      />
    );
  }
);

GenericString.displayName = `GenericString`;

export default GenericString;
