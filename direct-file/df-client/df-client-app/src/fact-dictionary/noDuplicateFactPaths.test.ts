import { wrappedFacts } from './generated/wrappedFacts.js';
import { describe, it } from 'vitest';

// Modules don't exist at runtime. On the FE these modules are combined within readRawFacts.ts. This means that if we
// have the same path in multiple modules one of them will be overwritten. To prevent this scenario we added this test.
describe(`The fact graph`, () => {
  const map = new Map<string, boolean>();
  const errors: string[] = [];
  it(`should not have any duplicate paths`, () => {
    for (const fact of wrappedFacts) {
      if (map.get(fact.path) === undefined || errors.indexOf(fact.path) !== -1) {
        map.set(fact.path, true);
      } else {
        errors.push(fact.path);
      }
    }
    if (errors.length) {
      throw new Error(`Duplicate fact found for ${errors.join(`, `)}`);
    }
  });
});
