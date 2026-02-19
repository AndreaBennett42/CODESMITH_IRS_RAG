import { DayFactory } from '@irs/js-factgraph-scala';
import { getFactStringLimit } from '../../../../factgraph/factLimitHelpers.js';
import { AbsolutePath } from '../../../../fact-dictionary/Path.js';

const DEFAULT_RETURN_VALUE = ``;

export function transformImportedDateWithLimit(
  input: string | null | undefined,
  factPath: AbsolutePath,
  defaultReturn: string | null = DEFAULT_RETURN_VALUE
): string | null {
  if (!input) {
    return defaultReturn;
  }
  const limit = getFactStringLimit(factPath, `Max`);
  const dateValidation = DayFactory(input, limit);
  return dateValidation.right ? input : defaultReturn;
}
