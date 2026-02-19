import { transformImportedStateOrProvence } from '../transformImportedStateOrProvence.js';

describe(`transformImportedStateOrProvence`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedStateOrProvence(null)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedStateOrProvence(``)).toBe(``);
  });

  it(`should return an empty string if input is less than two characters`, () => {
    expect(transformImportedStateOrProvence(`A`)).toBe(``);
  });

  it(`should return an empty string if input is more than two characters`, () => {
    expect(transformImportedStateOrProvence(`AAA`)).toBe(``);
  });
});
