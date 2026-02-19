import { Path } from '../fact-dictionary/Path.js';
import { wrappedFacts } from '../fact-dictionary/generated/wrappedFacts.js';

export type FactLimit = number | undefined;
export type LimitTypes = 'MaxLength' | 'Max' | 'Match' | 'Min';
export type FactStringLimit = string | undefined;

export const getFactIntLimit = (path: Path, limitType: LimitTypes): FactLimit | undefined => {
  const factDef = wrappedFacts.find((f) => f.path === path);
  const maybeFactLimit = factDef?.writable?.limits.find((l) => l.operation === limitType);
  return maybeFactLimit ? parseInt(maybeFactLimit.node.options.value) : undefined;
};

export const getFactFloatLimit = (path: Path, limitType: LimitTypes): FactLimit | undefined => {
  const factDef = wrappedFacts.find((f) => f.path === path);
  // eslint-disable-next-line eqeqeq
  const maybeFactLimit = factDef?.writable?.limits.find((l) => l.operation == limitType);
  return maybeFactLimit ? Math.round(parseFloat(maybeFactLimit.node.options.value) * 100) / 100 : undefined;
};

// Date limits are strings in the format 2020-10-10
export const getFactStringLimit = (path: Path, limitType: LimitTypes): FactStringLimit | undefined => {
  const factDef = wrappedFacts.find((f) => f.path === path);
  const maybeFactLimit = factDef?.writable?.limits.find((l) => l.operation === limitType);
  return maybeFactLimit ? maybeFactLimit.node.options.value : undefined;
};
