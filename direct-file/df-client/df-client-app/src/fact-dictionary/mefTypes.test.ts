import { describe, it, expect } from 'vitest';
import { facts } from './generated/facts.js';

// Java XML Parser No Likey
// The Java XML parser does its own thing and therefore we need to make sure
// not to use regexes that will be interpreted differenlty on the backend.
// This test checks that we avoid any known issues in the regexes.
describe(`Mef Types Regex Check`, () => {
  const mefFiltered = facts
    .filter((f) => {
      // eslint-disable-next-line eqeqeq
      return f.srcFile == `mefTypes.xml` && f[`@path`].endsWith(`Type`);
    })
    .map((f) => {
      const derived = f.Derived as { String: string };
      return { path: f[`@path`], regex: derived.String };
    });

  for (let i = 0; i < mefFiltered.length; i++) {
    const { path, regex } = mefFiltered[i];
    it(`${path} complies with regex rules`, () => {
      // Regex should not have leading space
      expect(regex).not.toMatch(/^ .*$/);
      // Regex should not have trailing space
      expect(regex).not.toMatch(/^.* $/);
    });
  }
});
