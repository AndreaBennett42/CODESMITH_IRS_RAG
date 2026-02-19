import { transformImportedStringWithLimit } from '../transformImportedStringWithLimit.js';

describe(`transformImportedStringWithLimit`, () => {
  describe(`for about you basic data`, () => {
    it(`should return an empty string if input is null or undefined`, () => {
      expect(transformImportedStringWithLimit(null, `/filers/*/firstName`)).toBe(``);
      expect(transformImportedStringWithLimit(null, `/filers/*/lastName`)).toBe(``);
    });

    it(`should return an empty string if input is an empty string`, () => {
      expect(transformImportedStringWithLimit(``, `/filers/*/firstName`)).toBe(``);
    });

    it(`should return an empty string if input contains special characters`, () => {
      expect(transformImportedStringWithLimit(`Zoë`, `/filers/*/firstName`)).toBe(``);
    });

    it(`should return a truncated string if input is over twenty characters`, () => {
      expect(transformImportedStringWithLimit(`Homer`, `/filers/*/firstName`)).toBe(`Homer`);
      expect(transformImportedStringWithLimit(`Simpson`, `/filers/*/lastName`)).toBe(`Simpson`);
    });
  });

  describe(`for w2 data`, () => {
    it(`should return a truncated string if input is over seventy-five characters for employer name`, () => {
      expect(
        transformImportedStringWithLimit(
          `Innovative Sustainable Solutions for Advanced Environmental Technologies Incorporated`,
          `/formW2s/*/employerName`
        )
      ).toBe(`Innovative Sustainable Solutions for Advanced Environmental Technologies In`);
    });
    it(`should allow for slashes in employer name line 2`, () => {
      expect(
        transformImportedStringWithLimit(
          `Global Premier Quality Assurance/Testing and Compliance Consulting Services Inc`,
          `/formW2s/*/writableEmployerNameLine2`
        )
      ).toBe(`Global Premier Quality Assurance/Testing and Compliance Consulting Services`);
    });
  });
});
