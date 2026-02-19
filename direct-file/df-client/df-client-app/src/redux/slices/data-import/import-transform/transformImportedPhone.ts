import { UsPhoneNumberFactory } from '@irs/js-factgraph-scala';

function isAllDigits(str: string): boolean {
  return /^[0-9]+$/.test(str);
}

export function transformImportedPhone(input: string | null | undefined): string {
  if (!input) {
    return ``;
  }
  const isInputAllDigits = isAllDigits(input);
  if (input.length === 10 && isInputAllDigits && passesPhoneValidation(`+1${input}`)) {
    return `+1${input}`;
  }
  if (input.length === 11 && input.startsWith(`1`) && isInputAllDigits && passesPhoneValidation(`+${input}`)) {
    return `+${input}`;
  }
  if (input.length === 12 && isAllDigits(input.slice(1)) && passesPhoneValidation(`+${input}`)) {
    return input;
  }
  return ``;
}

const passesPhoneValidation = (input: string): boolean => {
  const phoneValidation = UsPhoneNumberFactory(input);
  return phoneValidation.right ? true : false;
};
