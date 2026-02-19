import { getTranslatedLink, urlHasLanguagePlaceholder } from './urlUtils.js';

const URL_WITH_PLACEHOLDER = `www.directfile.gov/{LANGUAGE_CODE}/home/`;
const URL_WITH_QUERY_PARAM_PLACEHOLDER = `www.directfile.gov/home?lang={LANGUAGE_CODE}`;
const URL_WITHOUT_PLACEHOLDER = `www.directfile.gov/home/`;
const URL_WITH_MISSPELLED_PLACEHOLDER = `www.directfile.gov/{LANG_CODE}/home/`;

const URL_EN = `www.directfile.gov/en/home/`;
const URL_ES = `www.directfile.gov/es/home/`;
const URL_ENGLISH = `www.directfile.gov/home?lang=english`;

describe(`urlUtils`, () => {
  describe(urlHasLanguagePlaceholder.name, () => {
    it(`returns true if the placeholder is found`, () => {
      const hasPlaceholder = urlHasLanguagePlaceholder(URL_WITH_PLACEHOLDER);

      expect(hasPlaceholder).toBeTruthy();
    });

    it(`returns false if the placeholder is not present`, () => {
      const hasPlaceholder = urlHasLanguagePlaceholder(URL_WITHOUT_PLACEHOLDER);

      expect(hasPlaceholder).toBeFalsy();
    });

    it(`returns false if the placeholder is misspelled`, () => {
      const hasPlaceholder = urlHasLanguagePlaceholder(URL_WITH_MISSPELLED_PLACEHOLDER);

      expect(hasPlaceholder).toBeFalsy();
    });
  });

  describe(getTranslatedLink.name, () => {
    it(`gets the translated link using the shortened language code, en`, () => {
      const translatedLink = getTranslatedLink(URL_WITH_PLACEHOLDER, `en`);

      expect(translatedLink).toEqual(URL_EN);
    });

    it(`gets the translated link using the shortened language code, es`, () => {
      const translatedLink = getTranslatedLink(URL_WITH_PLACEHOLDER, `es`);

      expect(translatedLink).toEqual(URL_ES);
    });

    it(`gets the translated link using the longer language code, "english"`, () => {
      const translatedLink = getTranslatedLink(URL_WITH_QUERY_PARAM_PLACEHOLDER, `english`);

      expect(translatedLink).toEqual(URL_ENGLISH);
    });
  });
});
