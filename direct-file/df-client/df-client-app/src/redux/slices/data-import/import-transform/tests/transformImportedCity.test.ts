import { transformImportedCity } from '../transformImportedCity.js';

describe(`transformImportedCity`, () => {
  it(`should return an empty string if input is null or undefined`, () => {
    expect(transformImportedCity(null)).toBe(``);
    expect(transformImportedCity(undefined)).toBe(``);
  });

  it(`should return an empty string if input is an empty string`, () => {
    expect(transformImportedCity(``)).toBe(``);
  });

  it(`should return an empty string if input is less than minimum length`, () => {
    expect(transformImportedCity(`Al`)).toBe(``);
  });

  it(`should return an empty string if input is more than maximum length`, () => {
    expect(transformImportedCity(`Encantada-Ranchito-El Calaboz`)).toBe(``);
  });
});
