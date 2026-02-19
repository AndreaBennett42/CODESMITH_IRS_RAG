import { it, expect } from 'vitest';
import { wrappedFacts } from '../../fact-dictionary/generated/wrappedFacts.js';

it(`All enumoptions start with a slash`, () => {
  wrappedFacts
    .filter((f) => f.writable?.typeName === `Enum`)
    .forEach((enumFact) => {
      expect(
        enumFact?.writable?.options.optionsPath[0],
        `${enumFact.path} optionsPath needs to start with a slash`
      ).toBe(`/`);
    });
});
