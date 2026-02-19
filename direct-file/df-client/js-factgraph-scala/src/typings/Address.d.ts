import { ValidationFailure, ValidationResult } from './Validation';
import { ScalaMap } from './utils/conversionUtils';

export type Address = {
  streetAddress: string;
  city: string;
  postalCode: string;
  stateOrProvence: string;
  streetAddressLine2: string;
  country: string;
};
export type AddressFieldValidationFailure = ValidationFailure;
export interface AddressValidationFailure extends ValidationFailure {
  addressErrors: ScalaMap<keyof Address, AddressFieldValidationFailure>;
}
export type AddressResult = ValidationResult<Address, AddressValidationFailure>;

export declare function AddressFactory(
  streetAddress: string,
  city: string,
  postalCode: string,
  stateOrProvence: string,
  streetAddressLine2?: string,
  country?: string
): AddressResult;

export declare function formatAddressForHTML(addy: Address): string;
