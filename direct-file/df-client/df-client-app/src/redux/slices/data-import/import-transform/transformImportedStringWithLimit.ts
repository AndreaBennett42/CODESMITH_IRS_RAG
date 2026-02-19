import { StringFactory } from '@irs/js-factgraph-scala';
import { getFactIntLimit, getFactStringLimit } from '../../../../factgraph/factLimitHelpers.js';
import { AbsolutePath } from '../../../../fact-dictionary/Path.js';

const DEFAULT_RETURN_VALUE = ``;

export function transformImportedStringWithLimit(
  input: string | null | undefined,
  factPath: AbsolutePath,
  defaultReturn: string = DEFAULT_RETURN_VALUE
): string {
  if (!input) {
    return defaultReturn;
  }

  const regex = getFactStringLimit(factPath, `Match`);
  const stringPatternValidation = StringFactory(input, regex);
  if (stringPatternValidation.left) {
    return DEFAULT_RETURN_VALUE;
  }
  const intLimit = getFactIntLimit(factPath, `MaxLength`);
  if (intLimit && input.length > intLimit) {
    return input.substring(0, intLimit);
  }

  return stringPatternValidation.right ? input : defaultReturn;
}
