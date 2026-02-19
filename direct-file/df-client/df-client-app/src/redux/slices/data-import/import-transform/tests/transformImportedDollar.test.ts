import { transformImportedDollar } from '../transformImportedDollar.js';

describe(`transformImportedDollar`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedDollar(null)).toBe(``);
    expect(transformImportedDollar(undefined)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedDollar(``)).toBe(``);
  });

  it(`should return the dollar amount if valid`, () => {
    expect(transformImportedDollar(`20000`)).toBe(`20000`);
  });
});
