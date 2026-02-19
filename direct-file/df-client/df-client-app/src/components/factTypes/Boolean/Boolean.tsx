import { Radio, Checkbox } from '@trussworks/react-uswds';
import { forwardRef, useCallback, useEffect, useState } from 'react';

import { BooleanFactProps } from '../../../types/core.js';
import useFact from '../../../hooks/useFact.js';
import { ComplexFormControl } from '../../FormControl/index.js';
import Translation from '../../Translation/index.js';

const Boolean = forwardRef<HTMLInputElement, BooleanFactProps>(
  (
    {
      onValidData,
      showFeedback = false,
      concretePath,
      isValid,
      path,
      collectionId,
      inputType,
      i18nKeySuffixContext,
      readOnly = false,
      required = true,
    },
    ref
  ) => {
    const [fact, setFact] = useFact<boolean>(concretePath);

    let additionalTranslationSuffix = ``;
    if (i18nKeySuffixContext) {
      additionalTranslationSuffix = `.${i18nKeySuffixContext}`;
    }
    const maybeTranslationKeyYes = `fields.${path}${additionalTranslationSuffix}.boolean.yes`;
    const maybeTranslationKeyNo = `fields.${path}${additionalTranslationSuffix}.boolean.no`;

    const showError = showFeedback && !isValid;

    const [checkboxValue, setCheckboxValue] = useState(!!fact);

    const onCheckboxChange = useCallback(() => {
      // checkbox changing toggles the fact value
      setCheckboxValue((prev) => !prev);
    }, [setCheckboxValue]);

    useEffect(() => {
      if (inputType === `checkbox`) {
        setFact(checkboxValue);
        // If the input type is a checkbox, viewing the empty checkbox will qualify for setting the
        // fact to "false" unless the user toggles the checkbox
        required ? onValidData(concretePath, checkboxValue) : onValidData(concretePath, true);
      }
    }, [checkboxValue, setFact, onValidData, concretePath, inputType, required]);

    const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setFact(e.target.value === `Yes`);
      onValidData(concretePath, true);
    };

    return (
      <ComplexFormControl
        path={path}
        concretePath={concretePath}
        showError={showError}
        labelledBy={inputType === `checkbox` && !required ? `self` : `heading`}
        errorMessage={<Translation i18nKey={`enums.messages.requiredField`} collectionId={collectionId} />}
      >
        {inputType === `checkbox` ? (
          <Checkbox
            id={`id-${concretePath}-yes`}
            name={concretePath}
            label={<Translation i18nKey={[maybeTranslationKeyYes, `button.yes`]} collectionId={collectionId} />}
            value='Yes'
            onChange={onCheckboxChange}
            checked={checkboxValue}
            disabled={readOnly}
            required={required}
          />
        ) : (
          <>
            <Radio
              id={`id-${concretePath}-yes`}
              radioGroup={concretePath}
              name={concretePath}
              label={<Translation i18nKey={[maybeTranslationKeyYes, `button.yes`]} collectionId={collectionId} />}
              value='Yes'
              onChange={onChange}
              defaultChecked={fact}
              required
              disabled={readOnly}
            />
            <Radio
              id={`id-${concretePath}-no`}
              inputRef={ref}
              radioGroup={concretePath}
              name={concretePath}
              label={<Translation i18nKey={[maybeTranslationKeyNo, `button.no`]} collectionId={collectionId} />}
              value='No'
              onChange={onChange}
              // fact can be undefined, so we have to explicitly check for false
              defaultChecked={fact === false}
              required
              disabled={readOnly}
            />
          </>
        )}
      </ComplexFormControl>
    );
  }
);

Boolean.displayName = `Boolean`;

export default Boolean;
