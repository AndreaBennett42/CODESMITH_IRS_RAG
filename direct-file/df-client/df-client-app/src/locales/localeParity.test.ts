import rawEnLocale from '../locales/en.yaml';
import rawEsLocale from '../locales/es.yaml';
import Locale from '../scripts/locale.js';

const enLocale = new Locale(rawEnLocale);

const otherLocales = {
  es: new Locale(rawEsLocale),
  // Even though es is the only other language, we're future proofing the test
};

function keyIsExempt(key: string) {
  return key.startsWith(`articles.`) && null !== key.match(/_\w+$/g);
}

Object.entries(otherLocales).forEach(([lang, locale]) => {
  describe(`${lang} yaml`, () => {
    it(`has translations for every English key`, () => {
      for (const key of enLocale.keys) {
        if (!(keyIsExempt(key) || locale.has(key))) {
          throw new Error(`${key} was found in en but not in ${lang}`);
        }
      }
    });

    it(`doesn't have any keys not present in the English locale`, () => {
      for (const key of locale.keys) {
        if (!(enLocale.has(key) || keyIsExempt(key))) {
          throw new Error(`${key} was found in ${lang} but not in en`);
        }
      }
    });
  });
});
