import { ValidationFailure, ValidationResult } from './Validation';

export interface Pin {
  toString(): string;
}
export type PinValidationFailure = ValidationFailure;
export type PinResult = ValidationResult<Pin, PinValidationFailure>;

export declare function PinFactory(pin: string): PinResult;
