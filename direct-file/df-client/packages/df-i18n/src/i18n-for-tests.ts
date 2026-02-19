// eslint-disable-next-line no-restricted-imports
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

export function getI18nForTest() {
  // eslint-disable-next-line import/no-named-as-default-member
  i18n.use(initReactI18next).init({
    lng: `en`,
    fallbackLng: `en`,

    // have a common namespace used around the full app
    ns: [`test`, `translation`],
    defaultNS: `test`,

    keySeparator: `.`,

    debug: true,

    interpolation: {
      escapeValue: false, // not needed for react!!
    },

    resources: { en: { test: {} } },
  });
  return i18n;
}
