import { AbsolutePath } from '../../../../fact-dictionary/Path.js';
import { getFactFloatLimit } from '../../../../factgraph/factLimitHelpers.js';
import { DollarFactory } from '@irs/js-factgraph-scala';

export function transformImportedDollar(input: string | null | undefined, factPath?: AbsolutePath): string {
  if (!input) {
    return ``;
  }

  if (factPath) {
    const maxLimit = getFactFloatLimit(factPath, `Max`);
    const minLimit = getFactFloatLimit(factPath, `Min`);
    const dollarValidation = DollarFactory(input, maxLimit, minLimit);
    if (dollarValidation.right) {
      return input;
    }
  }

  const dollarValidation = DollarFactory(input);
  if (dollarValidation.right) {
    return input;
  }

  return ``;
}
