import { stripDisallowedCharacters } from '@irs/js-factgraph-scala';
import { getFactStringLimit } from '../../../../factgraph/factLimitHelpers.js';
import { AbsolutePath } from '../../../../fact-dictionary/Path.js';
import { transformImportedStringWithLimit } from './transformImportedStringWithLimit.js';

const DEFAULT_RETURN_VALUE = ``;

export function sanitizeStringWithLimit(
  input: string | null | undefined,
  factPath: AbsolutePath,
  defaultReturn: string = DEFAULT_RETURN_VALUE
): string {
  if (!input) {
    return defaultReturn;
  }

  const regex = getFactStringLimit(factPath, `Match`);
  const transformedInput = stripDisallowedCharacters(input, regex as string);
  return transformImportedStringWithLimit(transformedInput, factPath);
}
