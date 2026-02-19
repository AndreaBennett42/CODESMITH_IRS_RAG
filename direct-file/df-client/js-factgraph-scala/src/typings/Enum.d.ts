import { ValidationFailure, ValidationResult } from './Validation';

export interface Enum {
  getValue(): string;
}
export type EnumValidationFailure = ValidationFailure;
export type EnumResult = ValidationResult<Enum, EnumValidationFailure>;

export declare function EnumFactory(value: string, optionsPath: any): EnumResult;
