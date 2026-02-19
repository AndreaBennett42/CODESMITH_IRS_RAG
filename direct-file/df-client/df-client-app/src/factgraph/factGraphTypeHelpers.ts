import { wrappedFacts } from '../fact-dictionary/generated/wrappedFacts.js';
import { AbsolutePath } from '../fact-dictionary/Path.js';

export const FACT_PATH_TYPES = new Map<AbsolutePath, string>(
  wrappedFacts.map(({ path, derived, writable }) => [path, (derived ?? writable).typeName])
);
