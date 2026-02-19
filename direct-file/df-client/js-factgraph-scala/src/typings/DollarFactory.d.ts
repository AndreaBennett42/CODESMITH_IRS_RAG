import { ValidationFailure, ValidationResult } from './Validation';
import { FactLimit } from '../../../df-client-app/src/factgraph/factLimitHelpers';

export interface Dollar {}
export type DollarValidationFailure = ValidationFailure;
export type DollarResult = ValidationResult<Dollar, DollarValidationFailure>;

export declare function DollarFactory(dollar: string, maxLimit?: FactLimit, minLimit?: FactLimit): DollarResult;
