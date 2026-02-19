import { transformImportedPhone } from '../transformImportedPhone.js';

describe(`transformImportedPhone`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedPhone(null)).toBe(``);
    expect(transformImportedPhone(undefined)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedPhone(``)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedPhone(``)).toBe(``);
  });

  it(`should add a plus one`, () => {
    expect(transformImportedPhone(`2065551234`)).toBe(`+12065551234`);
  });

  it(`should add a one`, () => {
    expect(transformImportedPhone(`12065551234`)).toBe(`+12065551234`);
  });

  it(`should return nothing for a german phone`, () => {
    expect(transformImportedPhone(`+492284061240`)).toBe(``);
  });
});
