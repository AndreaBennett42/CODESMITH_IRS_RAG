import { paths } from './generated/paths.js';
import { describe, it, expect } from 'vitest';

describe(`Path naming`, () => {
  it(`All paths begin with slash`, () => {
    const factsThatDontStartWithSlash = paths.filter((p) => !p.startsWith(`/`));
    expect(factsThatDontStartWithSlash).toEqual([]);
  });
  it(`No path contains a tilda, since they are reserved for PDF psuedo-paths`, () => {
    const factsThatContainTilda = paths.filter((p) => p.includes(`~`));
    expect(factsThatContainTilda).toEqual([]);
  });
});
