import { ValidationFailure, ValidationResult } from './Validation';
import { ScalaSet } from './utils/conversionUtils';

export interface MultiEnum {
  getValue(): ScalaSet<string>;
}
export type MultiEnumValidationFailure = ValidationFailure;
export type MultiEnumResult = ValidationResult<MultiEnum, MultiEnumValidationFailure>;

export declare function MultiEnumFactory(value: ScalaSet<string>, optionsPath: any): MultiEnumResult;
