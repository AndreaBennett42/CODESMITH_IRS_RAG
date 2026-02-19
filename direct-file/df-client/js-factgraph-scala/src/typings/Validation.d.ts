import { either } from './utilsEither';

export interface UserFriendlyReason {
  toString(): string;
}

export interface ValidationFailureReason {
  toUserFriendlyReason(): UserFriendlyReason;
}

export interface ValidationFailure {
  validationMessage: ValidationFailureReason;
}

export type ValidationResult<T, ErrorType extends ValidationFailure> = either<ErrorType, T>;
