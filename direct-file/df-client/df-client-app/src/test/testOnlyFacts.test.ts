import { describe, it, expect } from 'vitest';
import { generateDependencyGraph } from '../fact-dictionary/generate-src/dependencyGraph.js';

describe(`The test only facts are used in an acceptable way`, () => {
  it(`the /todayOverride fact is only used as a part of /today`, () => {
    const dependencyMap = generateDependencyGraph();
    dependencyMap.forEach((deps, path) => {
      const paths = deps.map((p) => p.path);
      if (paths.includes(`/todayOverride`)) {
        expect(path).toBe(`/today`);
      }
    });
  });
});
