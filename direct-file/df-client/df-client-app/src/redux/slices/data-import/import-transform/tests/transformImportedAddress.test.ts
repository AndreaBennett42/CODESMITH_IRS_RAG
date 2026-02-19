import { transformImportedAddress } from '../transformImportedAddress.js';

describe(`transformImportedAddress`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedAddress(null)).toBe(``);
    expect(transformImportedAddress(undefined)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedAddress(``)).toBe(``);
  });

  it(`should preserve spaces`, () => {
    expect(transformImportedAddress(`123 456`)).toBe(`123 456`);
  });

  it(`should remove characters that are not allowed`, () => {
    expect(transformImportedAddress(`hello@world!`)).toBe(`helloworld`);
    expect(transformImportedAddress(`123 #$%^&*()abc`)).toBe(`123 abc`);
    expect(transformImportedAddress(`abc/xyz-123!`)).toBe(`abc/xyz-123`);
  });

  it(`should truncate the string to a maximum length of 35 characters`, () => {
    const longInput = `a`.repeat(50); // 50 characters
    expect(transformImportedAddress(longInput)).toBe(`a`.repeat(35)); // Should truncate to 35 characters

    const longInputWithAllowedChars = `!!abcdef/xyz-1234567890!!/xyz-1234567890`;
    // Should truncate to 35 characters
    expect(transformImportedAddress(longInputWithAllowedChars)).toBe(`abcdef/xyz-1234567890/xyz-123456789`);
  });

  it(`should return the input as-is if it is already valid and under 35 characters`, () => {
    expect(transformImportedAddress(`valid-input-123`)).toBe(`valid-input-123`);
    expect(transformImportedAddress(`abc/xyz-123`)).toBe(`abc/xyz-123`);
  });
});
