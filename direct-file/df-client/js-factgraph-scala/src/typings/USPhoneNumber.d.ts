import { ValidationFailure, ValidationResult } from './Validation';

export interface UsPhoneNumber {
  subscriberNumber: string;
  countryCode: `1`;
}
export type UsPhoneNumberValidationFailure = ValidationFailure;
export type UsPhoneNumberResult = ValidationResult<UsPhoneNumber, UsPhoneNumberValidationFailure>;

export declare function UsPhoneNumberFactory(phoneNumber: string): UsPhoneNumberResult;
