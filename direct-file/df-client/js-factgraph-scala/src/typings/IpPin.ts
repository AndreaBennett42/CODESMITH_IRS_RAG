import { ValidationFailure, ValidationResult } from './Validation';

export interface IpPin {
  toString(): string;
}
export type IpPinValidationFailure = ValidationFailure;
export type IpPinResult = ValidationResult<IpPin, IpPinValidationFailure>;

export declare function IpPinFactory(IpPin: string): IpPinResult;
