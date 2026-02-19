import { EinFactory } from '@irs/js-factgraph-scala';
import { stripNonNumeric } from '../../../../misc/misc.js';

const passesEinValidation = (input: string): boolean => {
  const einValidation = EinFactory(input);
  return einValidation.right ? true : false;
};

export function transformImportedEin(input: string | null | undefined): string {
  if (!input) {
    return ``;
  }
  const stripNonNumericFromInput = stripNonNumeric(input);
  if (passesEinValidation(stripNonNumericFromInput)) {
    return stripNonNumericFromInput;
  }
  return ``;
}
