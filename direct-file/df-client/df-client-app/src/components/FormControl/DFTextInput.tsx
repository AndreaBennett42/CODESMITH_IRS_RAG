import React from 'react';
import classnames from 'classnames';
export type ValidationStatus = 'error' | 'success';

type RequiredTextInputProps = {
  id: string;
  name: string;
  type: 'text' | 'email' | 'number' | 'password' | 'search' | 'tel' | 'url';
};

type CustomTextInputProps = {
  className?: string;
  validationStatus?: ValidationStatus;
  inputSize?: 'small' | 'medium';
  inputProps?: JSX.IntrinsicElements['input'];
  inputRef?: React.ForwardedRef<HTMLInputElement> | string;
};

export type OptionalTextInputProps = CustomTextInputProps & JSX.IntrinsicElements['input'];

export type TextInputProps = RequiredTextInputProps & OptionalTextInputProps;

export const DFTextInput = (props: TextInputProps): React.ReactElement => {
  const { id, name, type, className, validationStatus, inputSize, inputRef, ...inputProps } = props;

  const isError = validationStatus === `error`;
  const isSuccess = validationStatus === `success`;
  const isSmall = inputSize === `small`;
  const isMedium = inputSize === `medium`;

  const classes = classnames(
    `usa-input`,
    {
      'usa-input--error': isError,
      'usa-input--success': isSuccess,
      'usa-input--small': isSmall,
      'usa-input--medium': isMedium,
    },
    className
  );

  return (
    <input data-testid='textInput' className={classes} id={id} name={name} type={type} ref={inputRef} {...inputProps} />
  );
};

export default DFTextInput;
