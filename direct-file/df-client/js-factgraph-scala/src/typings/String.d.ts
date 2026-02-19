import { ValidationFailure, ValidationResult } from './Validation';

export interface LimitingString {
  toString(): string;
}

export type StringValidationFailure = ValidationFailure;
export type StringResult = ValidationResult<String, StringValidationFailure>;

export declare function StringFactory(s: string, pattern?: any): StringResult;

export declare function stripDisallowedCharacters(s: string, a: string): string;
