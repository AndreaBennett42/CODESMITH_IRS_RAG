import { ValidationFailure, ValidationResult } from './Validation';

export interface Tin {
  toString(): string;
}
export type TinValidationFailure = ValidationFailure;
export type TinResult = ValidationResult<Tin, TinValidationFailure>;

export declare function TinFactory(tin: string, allowAllZeros: boolean = false): TinResult;
