import { describe, it, expect } from 'vitest';
import { paths } from './generated/paths.js';
import { readMeFFactPaths } from './generate-src/readMeFFactPaths.js';
import { Path } from './Path.js';
import { generationFieldAliases } from './aliases.js';

describe(`Generated Fact Dictionary Code`, () => {
  it(`MeF file paths exist in the fact dictionary`, () => {
    // Added a new MeF fact and this test failed?
    // -- check that the fact-dictionary is up-to-date ^^
    // -- check if it is an aliased or extended path, and if it is add it
    //  to the corresponding generationFieldAliases list in `generate.ts`
    //  see: direct-file/df-client/fact-dictionary/src/generate-src/generate.ts
    const allPaths = new Set(paths);
    const mefPaths = readMeFFactPaths();
    const mefPathsNotInPathDictionary = mefPaths.filter((p) => !allPaths.has(p) && !p.includes(`~`));
    expect(
      mefPathsNotInPathDictionary.length,
      `MeF paths ${mefPathsNotInPathDictionary.join(`, `)} do not exist in fact dictionary`
    ).toBe(0);
  });

  it(`Generation field alias paths exist in the fact dictionary`, () => {
    const allPaths = new Set(paths);
    const aliasesNotInPathDictionary = generationFieldAliases.filter((p) => !allPaths.has(p as Path));
    expect(
      aliasesNotInPathDictionary.length,
      `Alias paths ${aliasesNotInPathDictionary.join(`, `)} do not exist in fact dictionary. 
      Maybe you need modified the fact dictionary and now need to update generate-src/generate.ts?`
    ).toBe(0);
  });
});
