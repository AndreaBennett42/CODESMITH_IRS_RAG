import { ValidationFailure, ValidationResult } from './Validation';
import { FactDateLimit } from '../../../df-client-app/src/factgraph/factLimitHelpers';

export interface Day {
  day: number;
  month: number;
  year: number;
}
export type DayValidationFailure = ValidationFailure;
export type DayResult = ValidationResult<Day, DayValidationFailure>;

export declare function DayFactory(date: string, maxLimit?: FactDateLimit): DayResult;
