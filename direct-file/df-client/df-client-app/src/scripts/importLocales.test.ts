import { describe, it, expect } from 'vitest';
import yaml from 'js-yaml';
import Locale, { translateKey, RawLocale } from './locale.js';

describe(`Locale`, () => {
  describe(`set with arrays`, () => {
    // If english contains an array
    const englishYaml = `
    body:
    - p : "The first thing"
    - p : "The second thing"
    - p : "The third thing"
    - p : "The fourth thing"
    `;

    it(`should create an array`, () => {
      // IF spanish has no body entry, it should properly create the array
      const spanishYaml = `
      helptext: help is on the way
      `;

      const enLocale = new Locale(yaml.load(englishYaml) as RawLocale);
      const langLocale = new Locale(yaml.load(spanishYaml) as RawLocale);

      // And translations provide values for 2 keys only
      const translations = [
        { key: `body.0.p`, lastKnownEn: `The first thing`, updatedEs: `The first spanish thing` },
        { key: `body.1.p`, lastKnownEn: `The second thing`, updatedEs: `The second spanish thing` },
      ];

      // Translate
      translations.forEach((row) => {
        translateKey(row.key, enLocale, enLocale.flatten(), langLocale, row.updatedEs, row.lastKnownEn);
      });

      // Expect output to contain spanish translation in an array
      const expectedYaml = `
      helptext: help is on the way
      body:
        - p: The first spanish thing
        - p: The second spanish thing
      `;
      const actualOutput = langLocale.toJSON();
      expect(actualOutput).toEqual(yaml.load(expectedYaml));
    });

    it(`should replace some but not all strings, correctly`, () => {
      // IF spanish contains an older array and we have some new translations
      const spanishYaml = `
      body:
      - p: old stuff
      - p: old stuff
      - p: old stuff
      - p: old stuff
      `;

      const enLocale = new Locale(yaml.load(englishYaml) as RawLocale);
      const langLocale = new Locale(yaml.load(spanishYaml) as RawLocale);

      // And translations provide values for 2 keys only
      const translations = [
        { key: `body.1.p`, lastKnownEn: `The second thing`, updatedEs: `The second spanish thing` },
        { key: `body.2.p`, lastKnownEn: `The third thing`, updatedEs: `The third spanish thing` },
      ];

      // Translate
      translations.forEach((row) => {
        translateKey(row.key, enLocale, enLocale.flatten(), langLocale, row.updatedEs, row.lastKnownEn);
      });

      // Expect output to contain spanish translation in an array
      // Expect 2nd and 3rd entries to be replaced, but not first and last
      const expectedYaml = `
      body:
        - p: old stuff
        - p: The second spanish thing
        - p: The third spanish thing
        - p: old stuff
      `;
      const actualOutput = langLocale.toJSON();
      expect(actualOutput).toEqual(yaml.load(expectedYaml));
    });

    it(`should replace non-arrays correctly`, () => {
      // And spanish contains a string
      const spanishYaml = `
      body:
        oldString: "Some stuff"
      `;

      const enLocale = new Locale(yaml.load(englishYaml) as RawLocale);
      const langLocale = new Locale(yaml.load(spanishYaml) as RawLocale);

      // And translations provide new values
      const translations = [
        { key: `body.0.p`, lastKnownEn: `The first thing`, updatedEs: `The first spanish thing` },
        { key: `body.1.p`, lastKnownEn: `The second thing`, updatedEs: `The second spanish thing` },
        { key: `body.2.p`, lastKnownEn: `The third thing`, updatedEs: `The third spanish thing` },
      ];

      // Translate
      translations.forEach((row) => {
        translateKey(row.key, enLocale, enLocale.flatten(), langLocale, row.updatedEs, row.lastKnownEn);
      });

      // Expect output to contain spanish translation in an array
      // Expect oldString to be removed
      const expectedYaml = `
      body:
        - p: The first spanish thing
        - p: The second spanish thing
        - p: The third spanish thing
      `;
      const actualOutput = langLocale.toJSON();
      expect(actualOutput).toEqual(yaml.load(expectedYaml));
    });
  });
});

describe(`Locale`, () => {
  it(`diff should return correct keys`, () => {
    const englishYaml = `
    europe:
      romania: bucharest
      portugal: lisbon
      greece: athens
      italy: venice
    asia:
    - vietnam: hanoi
    - thailand: chiang mai
    `;

    const spanishYaml = `
    europe:
      romania: cluj napoca
      italy: amalfi
    asia:
    - vietnam: dalat
    - thailand: chiang rai
    - mongolia: ulan batur
    `;

    const enLocale = new Locale(yaml.load(englishYaml) as RawLocale);
    const langLocale = new Locale(yaml.load(spanishYaml) as RawLocale);

    const diffResults = langLocale.diff(enLocale);
    expect(diffResults.thisCount).toBe(5);
    expect(diffResults.otherCount).toBe(6);

    const matchingKeys = diffResults.inBoth.map((tuple) => tuple[0]);
    expect(matchingKeys).toContain(`europe.italy`);
    expect(matchingKeys).toContain(`europe.romania`);
    expect(matchingKeys).toContain(`asia.0.vietnam`);
    expect(matchingKeys).toContain(`asia.1.thailand`);

    const missingKeys = diffResults.removed.map((tuple) => tuple[0]);
    expect(missingKeys).toContain(`europe.portugal`);
    expect(missingKeys).toContain(`europe.greece`);

    const addedKeys = diffResults.added.map((tuple) => tuple[0]);
    expect(addedKeys).toContain(`asia.2.mongolia`);
  });
});
