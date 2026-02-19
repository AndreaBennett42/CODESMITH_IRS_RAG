import { transformImportedEin } from '../transformImportedEin.js';

describe(`transformImportedEin`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedEin(null)).toBe(``);
    expect(transformImportedEin(undefined)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedEin(``)).toBe(``);
  });

  it(`should not return the EIN if input has disallowed prefix`, () => {
    expect(transformImportedEin(`079223332`)).toBe(``);
  });

  it(`should return an EIN if input is all numbers`, () => {
    expect(transformImportedEin(`009223332`)).toBe(`009223332`);
  });

  it(`should return an EIN if input has non-numeric characters`, () => {
    expect(transformImportedEin(`00-1234567`)).toBe(`001234567`);
  });
});
