import { transformImportedDateWithLimit } from '../transformImportedDateWithLimit.js';

describe(`transformImportedDateWithLimit`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedDateWithLimit(null, `/filers/*/dateOfBirth`)).toBe(``);
    expect(transformImportedDateWithLimit(undefined, `/filers/*/dateOfBirth`)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedDateWithLimit(``, `/filers/*/dateOfBirth`)).toBe(``);
  });

  it(`should return an empty string if input above max`, () => {
    expect(transformImportedDateWithLimit(`2050-01-01`, `/filers/*/dateOfBirth`)).toBe(``);
  });
});
