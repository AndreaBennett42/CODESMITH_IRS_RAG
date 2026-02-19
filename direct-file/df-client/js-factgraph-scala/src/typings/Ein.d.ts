import { ValidationFailure, ValidationResult } from './Validation';

export interface Ein {
  toString(): string;
}
export type EinValidationFailure = ValidationFailure;
export type EinResult = ValidationResult<Ein, EinValidationFailure>;

export declare function EinFactory(ein: string): EinResult;
