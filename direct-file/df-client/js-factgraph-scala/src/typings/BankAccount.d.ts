import { ValidationFailure, ValidationResult } from './Validation';

export type BankAccountType = `Checking` | `Savings`;
export interface BankAccount {
  accountType: BankAccountType;
  routingNumber: string;
  accountNumber: string;
}
export type BankAccountFieldValidationFailure = ValidationFailure;
export interface BankAccountValidationFailure extends ValidationFailure {
  fieldErrors: ScalaMap<keyof BankAccount, BankAccountValidationFailure>;
}
export type BankAccountResult = ValidationResult<BankAccount, BankAccountValidationFailure>;

export declare function BankAccountFactory(
  accountType: BankAccountType,
  routingNumber: string,
  accountNumber: string
): BankAccountResult;
