import { describe, it, expect } from 'vitest';
import { facts } from './generated/facts.js';
import readRawFacts from './generate-src/readRawFacts.js';

describe(`Fact generation`, () => {
  it(`Generates a fact for every path`, () => {
    const pathsInFacts = new Set<string>(facts.map((f) => f[`@path`]));
    // We have to use the raw paths from the fact dictionary
    // instead of our generated paths -- since our generated paths
    // contain aliases and other "not in the dictionary" style entries
    const rawPaths = readRawFacts().map((f) => f[`@path`]);
    for (const path of rawPaths) {
      if (!pathsInFacts.has(path)) {
        throw new Error(`No generated fact for ${path}`);
      }
    }
    // eslint-disable-next-line eqeqeq
    const duplicate = rawPaths.find((path, index) => rawPaths.indexOf(path) != index);
    if (duplicate) {
      throw new Error(`Had duplicate fact ${duplicate}`);
    }
    expect(pathsInFacts.size).toBe(rawPaths.length);
  });
});
