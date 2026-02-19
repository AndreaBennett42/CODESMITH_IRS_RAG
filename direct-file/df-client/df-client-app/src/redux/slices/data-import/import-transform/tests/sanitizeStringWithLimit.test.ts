import { sanitizeStringWithLimit } from '../sanitizeStringWithLimit.js';

describe(`sanitizeStringWithLimit`, () => {
  describe(`for w2s data`, () => {
    it(`should return an empty string if input is null or undefined`, () => {
      expect(sanitizeStringWithLimit(null, `/formW2s/*/employerName`)).toBe(``);
      expect(sanitizeStringWithLimit(undefined, `/formW2s/*/employerName`)).toBe(``);
    });

    it(`should return an empty string if input is an empty string`, () => {
      expect(sanitizeStringWithLimit(``, `/formW2s/*/employerName`)).toBe(``);
    });

    it(`should return a sanitized string if input contains special characters`, () => {
      expect(sanitizeStringWithLimit(`SPRINGFIELD, ATOMIC POWER AND LIGHT`, `/formW2s/*/employerName`)).toBe(
        `SPRINGFIELD ATOMIC POWER AND LIGHT`
      );
    });
  });
});
