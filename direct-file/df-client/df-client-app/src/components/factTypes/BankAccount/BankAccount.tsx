import { ChangeEvent, forwardRef, useCallback, useMemo, useState } from 'react';
import { FactProps } from '../../../types/core.js';
import {
  ComplexFormControl,
  FormFieldWrapper,
  buildControlErrorId,
  buildFormControlId,
  buildHintKey,
  buildHintId,
} from '../../FormControl/index.js';
import useFact from '../../../hooks/useFact.js';
import {
  type BankAccount as SFGBankAccount,
  type BankAccountValidationFailure,
  BankAccountFactory,
  scalaMapToJsMap,
} from '@irs/js-factgraph-scala';
import { useFactControl } from '../../../hooks/useFactControl.js';
import { Fieldset, Radio, TextInput } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import Translation from '../../Translation/index.js';
import HelpLink from '../../HelperText/HelpLink.js';

const BANK_ACCOUNT_TYPES = [`Checking`, `Savings`] as const;
export const BLANK_BANK_ACCOUNT = Object.freeze({
  accountType: `` as SFGBankAccount[`accountType`],
  routingNumber: ``,
  accountNumber: ``,
} as SFGBankAccount);

export const BankAccount = forwardRef<HTMLInputElement, FactProps>(
  ({ path, onValidData, showFeedback, isValid, concretePath, collectionId, readOnly }, ref) => {
    const [fact, setFact, clearFact] = useFact<SFGBankAccount>(concretePath);
    const [validationErrors, setValidationErrors] = useState(
      new Map<keyof SFGBankAccount, BankAccountValidationFailure>()
    );
    const { t, i18n } = useTranslation();

    const setValidity = useCallback(
      (isValid: boolean) => {
        onValidData(concretePath, isValid);
      },
      [concretePath, onValidData]
    );

    const onError = useCallback(
      (failureReason: BankAccountValidationFailure) => {
        setValidationErrors(scalaMapToJsMap(failureReason.fieldErrors));
        onValidData(concretePath, false);
      },
      [concretePath, onValidData]
    );

    const factParser = useCallback(
      ({ accountType, routingNumber, accountNumber }: SFGBankAccount) =>
        // Account number can contain letters, so we normalize to uppercase
        BankAccountFactory(accountType, routingNumber, accountNumber.toUpperCase()),
      []
    );

    const { rawValue, onChange } = useFactControl({
      setValidity,
      setFact,
      clearFact,
      isFactRequired: true, // Optional not implemented for Bank Accounts
      onError,
      factParser,
    });

    const spreadSafeFact: SFGBankAccount = useMemo(() => {
      // The BankAccount object from the fact graph doesn't support object spreading in an expected way
      // So we have to convert the fact to a plain object to make it safe to spread
      const { accountType, routingNumber, accountNumber } = fact ? fact : BLANK_BANK_ACCOUNT;
      return { accountType, routingNumber, accountNumber };
    }, [fact]);
    const currFields = rawValue ?? spreadSafeFact;
    const showError = showFeedback && !isValid;

    const onFieldChange = useCallback(
      (name: keyof SFGBankAccount, value: string) => {
        onChange({ ...currFields, [name]: value });
      },
      [currFields, onChange]
    );

    const getFieldErrorKey = useCallback(
      (name: keyof SFGBankAccount) => {
        const fieldValidationErrorCode = validationErrors
          .get(name)
          ?.validationMessage.toUserFriendlyReason()
          .toString();

        return currFields[name]
          ? `fields.${path}.errorMessages.${fieldValidationErrorCode}`
          : `enums.messages.requiredField`;
      },
      [currFields, path, validationErrors]
    );

    const buildCommonFieldProps = useCallback(
      (name: keyof SFGBankAccount) => {
        return {
          name: `${path}.${name}`,
          onChange: (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
            onFieldChange(name, event.target.value),
          required: true,
        } as const;
      },
      [onFieldChange, path]
    );

    const buildTextFieldProps = useCallback(
      (name: keyof SFGBankAccount) => {
        const controlId = buildFormControlId(concretePath, name);
        const errorId = showError && validationErrors.has(name) ? buildControlErrorId(controlId, name) : undefined;
        const hintKey = buildHintKey(path, name);
        const hintId = buildHintId(controlId);

        const fieldShowError = showError && validationErrors.has(name);
        const ids = [];
        if (fieldShowError) {
          ids.push(errorId);
        }
        if (i18n.exists(hintKey)) {
          ids.push(hintId);
        }
        return {
          ...buildCommonFieldProps(name),
          'data-testid': `field-${name}`,
          defaultValue: currFields[name],
          'aria-invalid': fieldShowError,
          validationStatus: fieldShowError ? (`error` as const) : undefined,
          id: controlId,
          'aria-describedby': ids.length > 0 ? ids.join(` `) : undefined,
        };
      },
      [buildCommonFieldProps, concretePath, currFields, path, showError, validationErrors, i18n]
    );

    const buildFieldWrapperProps = useCallback(
      (name: keyof SFGBankAccount) => {
        const fieldErrorKey = getFieldErrorKey(name);
        const controlId = buildFormControlId(concretePath, name);
        const hintId = buildHintId(controlId);

        return {
          labelKey: `fields.${path}.${name}`,
          path: path,
          name: name,
          collectionId,
          hintId: hintId,
          controlId: buildFormControlId(concretePath, name),
          showError: showError && (!currFields[name] || validationErrors.has(name)),
          errorMessage: <Translation i18nKey={fieldErrorKey} collectionId={collectionId} />,
        };
      },
      [collectionId, concretePath, currFields, getFieldErrorKey, path, showError, validationErrors]
    );

    const accountTypeControlId = buildFormControlId(concretePath, `accountType`);

    return (
      <ComplexFormControl
        path={path}
        concretePath={concretePath}
        showError={showError}
        errorMessage={t(`enums.messages.complexFactError`, { count: validationErrors.size })}
        labelledBy='self'
      >
        <Fieldset
          name={accountTypeControlId}
          id={accountTypeControlId}
          aria-invalid={showError && validationErrors.has(`accountType`)}
        >
          <FormFieldWrapper {...buildFieldWrapperProps(`accountType`)} useLegendAsLabel>
            {BANK_ACCOUNT_TYPES.map((option) => (
              <Radio
                {...buildCommonFieldProps(`accountType`)}
                data-testid={`field-accountType.${option}`}
                id={`${accountTypeControlId}.${option}`}
                radioGroup={accountTypeControlId}
                label={<Translation i18nKey={`fields.generics.accountType.${option}`} collectionId={collectionId} />}
                value={option}
                defaultChecked={option === currFields.accountType}
                key={option}
                disabled={readOnly}
              />
            ))}
          </FormFieldWrapper>
        </Fieldset>
        <FormFieldWrapper {...buildFieldWrapperProps(`routingNumber`)}>
          <TextInput inputRef={ref} type='text' {...buildTextFieldProps(`routingNumber`)} disabled={readOnly} />
        </FormFieldWrapper>
        <FormFieldWrapper {...buildFieldWrapperProps(`accountNumber`)}>
          <TextInput type='text' {...buildTextFieldProps(`accountNumber`)} disabled={readOnly} />
        </FormFieldWrapper>
        <HelpLink i18nKey={concretePath} collectionId={collectionId} />
      </ComplexFormControl>
    );
  }
);

BankAccount.displayName = `BankAccount`;
