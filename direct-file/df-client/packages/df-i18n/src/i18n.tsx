// eslint-disable-next-line no-restricted-imports
import i18next from 'i18next';

import { initReactI18next } from 'react-i18next';

// Get language with the following fallback
// 1. Language stored in localStorage
// 2. Language set in browser (although es → es-US)
// 3. Default `en`
export function getUserLang() {
  const storedLang = localStorage.getItem(`irs_df_language`);
  const browserLang = navigator.language;
  let selectedLang = `en`;
  if (!storedLang) {
    selectedLang = browserLang.startsWith(`es`) ? `es-US` : `en`;
    localStorage.setItem(`irs_df_language`, selectedLang);
  } else {
    selectedLang = storedLang;
  }
  return selectedLang;
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function commonInitI18n(resources: any) {
  // eslint-disable-next-line import/no-named-as-default-member
  const i18n = i18next.use(initReactI18next); // passes i18n down to react-i18next

  const selectedLang = getUserLang();
  document.documentElement.setAttribute(`lang`, selectedLang);

  await i18n.init({
    resources,
    lng: selectedLang,
    keySeparator: `.`,
    nsSeparator: false,
    interpolation: {
      escapeValue: false, // react already safes from xsst == `intlDate`) {
      // i18next doesn't provide support for default formatters so we are forced to roll our own
      alwaysFormat: true,
      format(value, format, lng, options) {
        if (!options) {
          return value;
        }

        const { interpolationkey, formatParams } = options;

        let baseResult;

        if (value instanceof Date) {
          baseResult = new Intl.DateTimeFormat(lng, formatParams ? formatParams[interpolationkey] : {}).format(value);
        } else if (typeof value == `number` && formatParams) {
          baseResult = new Intl.NumberFormat(lng, formatParams ? formatParams[interpolationkey] : {}).format(value);
        } else {
          baseResult = value;
        }

        let prefix = ``;

        if (format) {
          // i18next does not expose their default formatter so we cannot fallback to it
          switch (format) {
            case `prefixAnd`:
              if (typeof baseResult === `string`) {
                prefix = i18n.t(`articles.and`, { context: baseResult.toLocaleLowerCase().charAt(0) }) + ` `;
              }
              break;
            case `prefixOr`:
              if (typeof baseResult === `string`) {
                prefix = i18n.t(`articles.or`, { context: baseResult.toLocaleLowerCase().charAt(0) }) + ` `;
              }
              break;
            default:
              break;
          }
        }

        // Note: The format function needs to return undefined under certain circumstances, for unclear reasons
        return baseResult === undefined ? undefined : `${prefix}${baseResult}`;
      },
    },
  });
  return i18n;
}
