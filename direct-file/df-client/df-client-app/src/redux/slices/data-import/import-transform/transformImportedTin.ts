import { TinFactory } from '@irs/js-factgraph-scala';
import { stripNonNumeric } from '../../../../misc/misc.js';

const passesTinValidation = (input: string): boolean => {
  const tinValidation = TinFactory(input);
  return tinValidation.right ? true : false;
};

export function transformImportedTin(input: string | null | undefined): string {
  if (!input) {
    return ``;
  }
  const stripNonNumericFromInput = stripNonNumeric(input);
  if (passesTinValidation(stripNonNumericFromInput)) {
    return stripNonNumericFromInput;
  }
  return ``;
}
